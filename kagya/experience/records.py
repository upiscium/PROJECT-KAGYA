"""Immutable, reference-first Experience contracts for R12 U1.

This module deliberately contains no persistence or runtime integration.  It
only validates the typed evidence that a later Memory-owned Experience
participant may publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import Final

from kagya.body import EmotionState, EmotionUpdate
from kagya.cognition import (
    AppraisalResult,
    LossMeasurement,
)
from kagya.identifiers import validate_identifier


EXPERIENCE_SCHEMA_VERSION: Final = 1
EXPERIENCE_SALIENCE_VERSION: Final = 1
EXPERIENCE_MAX_REVISIONS: Final = 32
EXPERIENCE_MAX_REVISION: Final = 2**31 - 1
EXPERIENCE_MAX_EVENT_SEQUENCE: Final = 2**63 - 1
EXPERIENCE_MAX_REVISION_EVIDENCE: Final = 32
_EXPERIENCE_DOMAIN: Final = b"PROJECT-KAGYA:R12:EXPERIENCE:V1\0"
_EXPERIENCE_REVISION_DOMAIN: Final = b"PROJECT-KAGYA:R12:EXPERIENCE-REVISION:V1\0"
_MODEL_KEY_PATTERN: Final = re.compile(r"model\.[0-9a-f]{64}\Z")


class ExperienceLifecycle(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class ExperienceRevisionOperation(str, Enum):
    REASSESS = "reassess"
    CORRECT = "correct"
    SUPERSEDE = "supersede"
    RETRACT = "retract"


class ExperienceRevisionReason(str, Enum):
    REASSESSMENT = "reassessment"
    CORRECTION = "correction"
    SUPERSESSION = "supersession"
    RETRACTION = "retraction"


class ExperienceMeasurementInvalidReason(str, Enum):
    """Frozen R10 measurement-invalid vocabulary stored by Experience V1."""

    EMPTY_TARGET = "empty_target"
    PROVIDER_ERROR = "provider_error"
    NON_FINITE_LOSS = "non_finite_loss"


class ExperienceAppraisalReasonCode(str, Enum):
    """Frozen R10 appraisal-reason vocabulary stored by Experience V1."""

    NOVELTY_MEASURED = "novelty_measured"
    NOVELTY_INVALID = "novelty_invalid"
    GOAL_PROGRESS = "goal_progress"
    GOAL_SETBACK = "goal_setback"
    THREAT = "threat"


class ExperienceEmotionUpdateReasonCode(str, Enum):
    """Frozen R10 emotion-update vocabulary stored by Experience V1."""

    APPRAISAL_APPLIED = "appraisal_applied"
    NOVELTY_OMITTED = "novelty_omitted"
    TIME_RECOVERY = "time_recovery"
    TIMELINE_INITIALIZED = "timeline_initialized"
    NO_MATERIAL_APPRAISAL = "no_material_appraisal"


ExperienceLossInvalidReason = ExperienceMeasurementInvalidReason
ExperienceEmotionReasonCode = ExperienceEmotionUpdateReasonCode


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__}")
    return value


def _finite(value: object, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return result


def _nonnegative_int(
    value: object,
    name: str,
    *,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative exact integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds its bound")
    return value


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    try:
        return validate_identifier(value)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{name} must be a valid identifier") from error


def _canonical_references(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a non-empty tuple")
    if len(value) > EXPERIENCE_MAX_REVISION_EVIDENCE:
        raise ValueError(f"{name} exceeds its bound")
    references = tuple(validate_identifier(item) for item in value)
    if references != tuple(sorted(set(references))):
        raise ValueError(f"{name} must be sorted and unique")
    return references


def _positive_int(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a positive bounded exact integer")
    return value


def _utc_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value.astimezone(timezone.utc)


def _model_key(value: object) -> str:
    if type(value) is not str or _MODEL_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError("model_key must be an opaque model key")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reasons(
    value: object,
    enum_type: type[Enum],
    name: str,
) -> tuple[Enum, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if any(type(item) is not enum_type for item in value):
        raise TypeError(f"{name} contains an invalid closed reason")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _datetime_value(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True, slots=True)
class ExperienceMeasurementEvidence:
    """Reference-first provider measurement evidence.

    Invalid novelty remains ``None``.  It is never represented as a numeric
    zero, because zero is a valid measured value with different meaning.
    """

    model_key: str
    valid: bool
    invalid_reason: ExperienceMeasurementInvalidReason | None = None
    calibrated_novelty: float | None = None

    def __post_init__(self) -> None:
        _model_key(self.model_key)
        if type(self.valid) is not bool:
            raise TypeError("valid must be bool")
        if self.valid:
            if self.invalid_reason is not None:
                raise ValueError("valid evidence cannot have an invalid reason")
            novelty = _finite(
                self.calibrated_novelty,
                "calibrated_novelty",
                lower=0.0,
                upper=1.0,
            )
            object.__setattr__(self, "calibrated_novelty", novelty)
        else:
            if type(self.invalid_reason) is not ExperienceMeasurementInvalidReason:
                raise TypeError("invalid evidence requires a closed invalid reason")
            if self.calibrated_novelty is not None:
                raise ValueError("invalid evidence cannot carry novelty")

    @classmethod
    def from_loss_measurement(cls, measurement: LossMeasurement) -> ExperienceMeasurementEvidence:
        if not isinstance(measurement, LossMeasurement):
            raise TypeError("measurement must be LossMeasurement")
        invalid_reason = None
        if measurement.invalid_reason is not None:
            try:
                invalid_reason = ExperienceMeasurementInvalidReason(
                    measurement.invalid_reason.value
                )
            except (AttributeError, ValueError) as error:
                raise ValueError("unsupported R10 measurement invalid reason") from error
        return cls(
            model_key=measurement.model_key,
            valid=measurement.valid,
            invalid_reason=invalid_reason,
            calibrated_novelty=measurement.calibrated_novelty,
        )

    from_measurement = from_loss_measurement


@dataclass(frozen=True, slots=True)
class ExperienceAppraisalEvidence:
    """The bounded R10 appraisal result copied without inventing values."""

    novelty: float | None
    novelty_valid: bool
    goal_progress: float | None = None
    threat: float | None = None
    controllability: float | None = None
    certainty: float | None = None
    social_relevance: float | None = None
    effort_cost: float | None = None
    reason_codes: tuple[ExperienceAppraisalReasonCode, ...] = ()

    def __post_init__(self) -> None:
        if type(self.novelty_valid) is not bool:
            raise TypeError("novelty_valid must be bool")
        novelty = None if self.novelty is None else _finite(
            self.novelty, "novelty", lower=0.0, upper=1.0
        )
        if self.novelty_valid != (novelty is not None):
            raise ValueError("novelty and novelty_valid disagree")
        object.__setattr__(self, "novelty", novelty)
        for name in (
            "goal_progress",
            "threat",
            "controllability",
            "certainty",
            "social_relevance",
            "effort_cost",
        ):
            lower, upper = (-1.0, 1.0) if name == "goal_progress" else (0.0, 1.0)
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                None if value is None else _finite(value, name, lower=lower, upper=upper),
            )
        _reasons(self.reason_codes, ExperienceAppraisalReasonCode, "reason_codes")

    @property
    def reasons(self) -> tuple[ExperienceAppraisalReasonCode, ...]:
        return self.reason_codes

    @classmethod
    def from_result(cls, result: AppraisalResult) -> ExperienceAppraisalEvidence:
        if not isinstance(result, AppraisalResult):
            raise TypeError("result must be AppraisalResult")
        try:
            reason_codes = tuple(
                ExperienceAppraisalReasonCode(reason.value) for reason in result.reasons
            )
        except (AttributeError, ValueError) as error:
            raise ValueError("unsupported R10 appraisal reason") from error
        return cls(
            novelty=result.novelty,
            novelty_valid=result.novelty_valid,
            goal_progress=result.goal_progress,
            threat=result.threat,
            controllability=result.controllability,
            certainty=result.certainty,
            social_relevance=result.social_relevance,
            effort_cost=result.effort_cost,
            reason_codes=reason_codes,
        )


@dataclass(frozen=True, slots=True)
class ExperienceEmotionProjection:
    """Bounded valence/arousal state copied from R10."""

    valence: float
    arousal: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "valence", _finite(self.valence, "valence", lower=-1.0, upper=1.0)
        )
        object.__setattr__(
            self, "arousal", _finite(self.arousal, "arousal", lower=0.0, upper=1.0)
        )

    @classmethod
    def from_state(cls, state: EmotionState) -> ExperienceEmotionProjection:
        if not isinstance(state, EmotionState):
            raise TypeError("state must be EmotionState")
        return cls(state.valence, state.arousal)


@dataclass(frozen=True, slots=True)
class ExperienceValenceContributions:
    """Frozen R10 V1 valence contribution projection."""

    goal_progress: float = 0.0
    threat: float = 0.0
    effort_cost: float = 0.0
    controllability: float = 0.0

    def __post_init__(self) -> None:
        bounds = {
            "goal_progress": (-0.7, 0.7),
            "threat": (-0.8, 0.0),
            "effort_cost": (-0.3, 0.0),
            "controllability": (-0.1, 0.1),
        }
        for name, (lower, upper) in bounds.items():
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, lower=lower, upper=upper),
            )


@dataclass(frozen=True, slots=True)
class ExperienceArousalContributions:
    """Frozen R10 V1 arousal contribution projection."""

    novelty: float = 0.0
    threat: float = 0.0
    effort_cost: float = 0.0
    social_relevance: float = 0.0
    uncertainty: float = 0.0
    low_controllability: float = 0.0

    def __post_init__(self) -> None:
        bounds = {
            "novelty": (0.0, 0.6),
            "threat": (0.0, 0.7),
            "effort_cost": (0.0, 0.3),
            "social_relevance": (0.0, 0.2),
            "uncertainty": (0.0, 0.2),
            "low_controllability": (0.0, 0.2),
        }
        for name, (lower, upper) in bounds.items():
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, lower=lower, upper=upper),
            )


@dataclass(frozen=True, slots=True)
class ExperienceEmotionContributions:
    """R12-owned contribution vectors frozen from the current R10 contract."""

    valence_contributions: ExperienceValenceContributions = field(
        default_factory=ExperienceValenceContributions
    )
    arousal_contributions: ExperienceArousalContributions = field(
        default_factory=ExperienceArousalContributions
    )

    def __post_init__(self) -> None:
        if not isinstance(self.valence_contributions, ExperienceValenceContributions):
            raise TypeError("valence_contributions must be ExperienceValenceContributions")
        if not isinstance(self.arousal_contributions, ExperienceArousalContributions):
            raise TypeError("arousal_contributions must be ExperienceArousalContributions")

    @property
    def valence(self) -> ExperienceValenceContributions:
        return self.valence_contributions

    @property
    def arousal(self) -> ExperienceArousalContributions:
        return self.arousal_contributions

    @classmethod
    def from_update(cls, update: EmotionUpdate) -> ExperienceEmotionContributions:
        if not isinstance(update, EmotionUpdate):
            raise TypeError("update must be EmotionUpdate")
        return cls(
            ExperienceValenceContributions(
                goal_progress=update.valence_contributions.goal_progress,
                threat=update.valence_contributions.threat,
                effort_cost=update.valence_contributions.effort_cost,
                controllability=update.valence_contributions.controllability,
            ),
            ExperienceArousalContributions(
                novelty=update.arousal_contributions.novelty,
                threat=update.arousal_contributions.threat,
                effort_cost=update.arousal_contributions.effort_cost,
                social_relevance=update.arousal_contributions.social_relevance,
                uncertainty=update.arousal_contributions.uncertainty,
                low_controllability=update.arousal_contributions.low_controllability,
            ),
        )


def calculate_subjective_salience(
    measurement: ExperienceMeasurementEvidence,
    pre_appraisal_emotion: ExperienceEmotionProjection,
    post_appraisal_emotion: ExperienceEmotionProjection,
) -> float:
    """Calculate the fixed R12 V1 deterministic salience projection."""

    if not isinstance(measurement, ExperienceMeasurementEvidence):
        raise TypeError("measurement must be ExperienceMeasurementEvidence")
    if not isinstance(pre_appraisal_emotion, ExperienceEmotionProjection):
        raise TypeError("pre_appraisal_emotion must be ExperienceEmotionProjection")
    if not isinstance(post_appraisal_emotion, ExperienceEmotionProjection):
        raise TypeError("post_appraisal_emotion must be ExperienceEmotionProjection")
    affective_impact = max(
        abs(post_appraisal_emotion.valence - pre_appraisal_emotion.valence) / 2.0,
        abs(post_appraisal_emotion.arousal - pre_appraisal_emotion.arousal),
    )
    if measurement.valid:
        assert measurement.calibrated_novelty is not None
        result = (measurement.calibrated_novelty + affective_impact) / 2.0
    else:
        result = affective_impact
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("subjective salience must be finite and in [0, 1]")
    return result


compute_subjective_salience = calculate_subjective_salience
subjective_salience = calculate_subjective_salience


def _revision_fields(record: ExperienceRevisionRecord) -> dict[str, object]:
    return {
        "created_at": _datetime_value(record.created_at),
        "event_id": record.event_id,
        "event_sequence": record.event_sequence,
        "evidence_refs": list(record.evidence_refs),
        "experience_id": record.experience_id,
        "operation": record.operation.value,
        "previous_revision_digest": record.previous_revision_digest,
        "reason": record.reason.value,
        "revision": record.revision,
    }


def canonical_experience_revision_payload(record: ExperienceRevisionRecord) -> bytes:
    if not isinstance(record, ExperienceRevisionRecord):
        raise TypeError("record must be ExperienceRevisionRecord")
    return _EXPERIENCE_REVISION_DOMAIN + _canonical_json(_revision_fields(record))


def experience_revision_digest(record: ExperienceRevisionRecord) -> str:
    return hashlib.sha256(canonical_experience_revision_payload(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class ExperienceRevisionRecord:
    experience_id: str
    revision: int
    operation: ExperienceRevisionOperation
    reason: ExperienceRevisionReason
    created_at: datetime
    evidence_refs: tuple[str, ...] = ()
    previous_revision_digest: str | None = None
    event_id: str | None = None
    event_sequence: int | None = None
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        validate_identifier(self.experience_id)
        _nonnegative_int(
            self.revision,
            "revision",
            maximum=EXPERIENCE_MAX_REVISION,
        )
        _enum(self.operation, ExperienceRevisionOperation, "operation")
        _enum(self.reason, ExperienceRevisionReason, "reason")
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        object.__setattr__(
            self,
            "evidence_refs",
            _canonical_references(self.evidence_refs, "evidence_refs"),
        )
        if (self.event_id is None) != (self.event_sequence is None):
            raise ValueError("event_id and event_sequence must be supplied together")
        object.__setattr__(self, "event_id", _optional_identifier(self.event_id, "event_id"))
        if self.event_sequence is not None:
            object.__setattr__(
                self,
                "event_sequence",
                _positive_int(
                    self.event_sequence,
                    "event_sequence",
                    maximum=EXPERIENCE_MAX_EVENT_SEQUENCE,
                ),
            )
        if self.revision == 0 and self.previous_revision_digest is not None:
            raise ValueError("genesis revision cannot have a previous digest")
        if self.revision > 0 and self.previous_revision_digest is None:
            raise ValueError("non-genesis revision requires a previous digest")
        if self.previous_revision_digest is not None:
            _digest(self.previous_revision_digest, "previous_revision_digest")
        object.__setattr__(self, "record_digest", experience_revision_digest(self))


def _emotion_reasons(
    value: tuple[ExperienceEmotionUpdateReasonCode, ...], name: str
) -> tuple[ExperienceEmotionUpdateReasonCode, ...]:
    result = _reasons(value, ExperienceEmotionUpdateReasonCode, name)
    return result  # type: ignore[return-value]


def _record_fields(record: ExperienceRecord) -> dict[str, object]:
    def contribution_fields(contributions: ExperienceEmotionContributions) -> dict[str, object]:
        return {
            "arousal": {
                name: getattr(contributions.arousal_contributions, name).hex()
                for name in (
                    "novelty",
                    "threat",
                    "effort_cost",
                    "social_relevance",
                    "uncertainty",
                    "low_controllability",
                )
            },
            "valence": {
                name: getattr(contributions.valence_contributions, name).hex()
                for name in (
                    "goal_progress",
                    "threat",
                    "effort_cost",
                    "controllability",
                )
            },
        }

    def emotion_fields(emotion: ExperienceEmotionProjection) -> dict[str, str]:
        return {"arousal": emotion.arousal.hex(), "valence": emotion.valence.hex()}

    def appraisal_fields(appraisal: ExperienceAppraisalEvidence) -> dict[str, object]:
        return {
            "certainty": None if appraisal.certainty is None else appraisal.certainty.hex(),
            "effort_cost": None
            if appraisal.effort_cost is None
            else appraisal.effort_cost.hex(),
            "goal_progress": None
            if appraisal.goal_progress is None
            else appraisal.goal_progress.hex(),
            "novelty": None if appraisal.novelty is None else appraisal.novelty.hex(),
            "novelty_valid": appraisal.novelty_valid,
            "reason_codes": [reason.value for reason in appraisal.reason_codes],
            "social_relevance": None
            if appraisal.social_relevance is None
            else appraisal.social_relevance.hex(),
            "threat": None if appraisal.threat is None else appraisal.threat.hex(),
            "controllability": None
            if appraisal.controllability is None
            else appraisal.controllability.hex(),
        }

    return {
        "appraisal": appraisal_fields(record.appraisal),
        "context_id": record.context_id,
        "created_at": _datetime_value(record.created_at),
        "emotion_contributions": contribution_fields(record.emotion_contributions),
        "emotion_update_reasons": [
            reason.value for reason in record.emotion_update_reasons
        ],
        "experience_id": record.experience_id,
        "history_anchor_digest": record.history_anchor_digest,
        "lifecycle": record.lifecycle.value,
        "measurement": {
            "calibrated_novelty": None
            if record.measurement.calibrated_novelty is None
            else record.measurement.calibrated_novelty.hex(),
            "invalid_reason": None
            if record.measurement.invalid_reason is None
            else record.measurement.invalid_reason.value,
            "model_key": record.measurement.model_key,
            "valid": record.measurement.valid,
        },
        "post_appraisal_emotion": emotion_fields(record.post_appraisal_emotion),
        "pre_appraisal_emotion": emotion_fields(record.pre_appraisal_emotion),
        "revision": record.revision,
        "revision_history": [
            revision.record_digest for revision in record.revision_history
        ],
        "schema_version": record.schema_version,
        "source_episode_id": record.source_episode_id,
        "source_event_id": record.source_event_id,
        "source_event_sequence": record.source_event_sequence,
        "subjective_salience": record.subjective_salience.hex(),
        "superseded_by_id": record.superseded_by_id,
        "temporal_update_reasons": [
            reason.value for reason in record.temporal_update_reasons
        ],
    }


def canonical_experience_payload(record: ExperienceRecord) -> bytes:
    if not isinstance(record, ExperienceRecord):
        raise TypeError("record must be ExperienceRecord")
    return _EXPERIENCE_DOMAIN + _canonical_json(_record_fields(record))


def experience_record_digest(record: ExperienceRecord) -> str:
    return hashlib.sha256(canonical_experience_payload(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    experience_id: str
    revision: int
    lifecycle: ExperienceLifecycle
    source_event_id: str
    source_event_sequence: int
    source_episode_id: str
    context_id: str
    measurement: ExperienceMeasurementEvidence
    appraisal: ExperienceAppraisalEvidence
    pre_appraisal_emotion: ExperienceEmotionProjection
    temporal_update_reasons: tuple[ExperienceEmotionUpdateReasonCode, ...]
    post_appraisal_emotion: ExperienceEmotionProjection
    emotion_contributions: ExperienceEmotionContributions
    emotion_update_reasons: tuple[ExperienceEmotionUpdateReasonCode, ...]
    subjective_salience: float
    created_at: datetime
    schema_version: int = EXPERIENCE_SCHEMA_VERSION
    superseded_by_id: str | None = None
    revision_history: tuple[ExperienceRevisionRecord, ...] = ()
    history_anchor_digest: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.experience_id)
        if type(self.schema_version) is not int or self.schema_version != EXPERIENCE_SCHEMA_VERSION:
            raise ValueError("unsupported Experience schema version")
        _nonnegative_int(
            self.revision,
            "revision",
            maximum=EXPERIENCE_MAX_REVISION,
        )
        _enum(self.lifecycle, ExperienceLifecycle, "lifecycle")
        validate_identifier(self.source_event_id)
        _nonnegative_int(
            self.source_event_sequence,
            "source_event_sequence",
            maximum=EXPERIENCE_MAX_EVENT_SEQUENCE,
        )
        validate_identifier(self.source_episode_id)
        object.__setattr__(self, "context_id", validate_identifier(self.context_id))
        for name, expected in (
            ("measurement", ExperienceMeasurementEvidence),
            ("appraisal", ExperienceAppraisalEvidence),
            ("pre_appraisal_emotion", ExperienceEmotionProjection),
            ("post_appraisal_emotion", ExperienceEmotionProjection),
            ("emotion_contributions", ExperienceEmotionContributions),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        object.__setattr__(
            self,
            "temporal_update_reasons",
            _emotion_reasons(self.temporal_update_reasons, "temporal_update_reasons"),
        )
        object.__setattr__(
            self,
            "emotion_update_reasons",
            _emotion_reasons(self.emotion_update_reasons, "emotion_update_reasons"),
        )
        object.__setattr__(
            self,
            "subjective_salience",
            _finite(self.subjective_salience, "subjective_salience", lower=0.0, upper=1.0),
        )
        expected_salience = calculate_subjective_salience(
            self.measurement,
            self.pre_appraisal_emotion,
            self.post_appraisal_emotion,
        )
        if self.subjective_salience != expected_salience:
            raise ValueError("subjective_salience does not match the V1 projection")
        object.__setattr__(self, "created_at", _utc_datetime(self.created_at, "created_at"))
        object.__setattr__(
            self,
            "superseded_by_id",
            _optional_identifier(self.superseded_by_id, "superseded_by_id"),
        )
        if self.lifecycle is ExperienceLifecycle.SUPERSEDED:
            if self.superseded_by_id is None:
                raise ValueError("superseded Experiences require superseded_by_id")
        elif self.superseded_by_id is not None:
            raise ValueError("only superseded Experiences may name a successor")
        if type(self.revision_history) is not tuple:
            raise TypeError("revision_history must be a tuple")
        if len(self.revision_history) > EXPERIENCE_MAX_REVISIONS:
            raise ValueError("Experience revision history exceeds its bound")
        if self.revision == 0 and self.revision_history:
            raise ValueError("revision zero cannot retain prior revisions")
        if self.revision == 0 and self.history_anchor_digest is not None:
            raise ValueError("revision zero cannot retain a history anchor")
        if self.revision > 0 and not self.revision_history and self.history_anchor_digest is None:
            raise ValueError("nonzero revision requires history or an anchor")
        previous: ExperienceRevisionRecord | None = None
        for item in self.revision_history:
            if not isinstance(item, ExperienceRevisionRecord):
                raise TypeError("revision_history contains an invalid record")
            if item.experience_id != self.experience_id:
                raise ValueError("revision history cannot mix Experience IDs")
            if previous is not None:
                if item.revision != previous.revision + 1:
                    raise ValueError("Experience revision history is non-monotonic")
                if item.previous_revision_digest != previous.record_digest:
                    raise ValueError("Experience revision history has a broken digest link")
            elif item.revision > 0:
                if self.history_anchor_digest is None:
                    raise ValueError("first retained revision lacks an anchor")
                if item.previous_revision_digest != self.history_anchor_digest:
                    raise ValueError("first retained revision does not link to its anchor")
            elif item.previous_revision_digest is not None:
                raise ValueError("genesis revision cannot link to a prior digest")
            if item.revision > self.revision:
                raise ValueError("Experience revision history contains a future revision")
            previous = item
        if previous is not None and previous.revision != self.revision - 1:
            raise ValueError("Experience revision history does not reach the current revision")
        if self.history_anchor_digest is not None:
            _digest(self.history_anchor_digest, "history_anchor_digest")
