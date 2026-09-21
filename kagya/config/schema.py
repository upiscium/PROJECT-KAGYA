"""Typed configuration schema for PROJECT-KAGYA."""

from pathlib import Path
import math

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from kagya.identifiers import validate_identifier


class StrictBaseModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class ProjectSettings(StrictBaseModel):
    name: str
    environment: str


class ModelSettings(StrictBaseModel):
    primary_id: str = Field(min_length=1)
    fallback_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    device: str
    dtype: str
    load_in_4bit: bool


class GenerationSettings(StrictBaseModel):
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    do_sample: bool


class EmotionSettings(StrictBaseModel):
    baseline_surprisal: float = Field(ge=0.0)
    high_emotion_threshold: float = Field(ge=0.0, le=1.0)
    decay_rate: float = Field(ge=0.0)
    timer_enabled: bool = False
    timer_interval_seconds: float = Field(default=60.0, gt=0.0)
    appraisal_response_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    resting_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    resting_arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    valence_recovery_rate: float = Field(default=0.01, ge=0.0)
    arousal_recovery_rate: float = Field(default=0.02, ge=0.0)

    @field_validator(
        "baseline_surprisal",
        "high_emotion_threshold",
        "decay_rate",
        "timer_interval_seconds",
        "appraisal_response_rate",
        "resting_valence",
        "resting_arousal",
        "valence_recovery_rate",
        "arousal_recovery_rate",
    )
    @classmethod
    def require_finite_emotion_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("emotion settings must be finite")
        return value


class AppraisalSettings(StrictBaseModel):
    initial_loss_scale: float = Field(default=1.0, gt=0.0)
    minimum_loss_scale: float = Field(default=0.01, gt=0.0)

    @field_validator("initial_loss_scale", "minimum_loss_scale")
    @classmethod
    def require_finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("loss calibration scales must be finite")
        return value


class MemorySettings(StrictBaseModel):
    persist_directory: Path
    db1_collection: str = Field(min_length=1)
    db2_collection: str = Field(min_length=1)
    db1_top_k: int = Field(gt=0)
    db2_top_k: int = Field(gt=0)
    embedding_model_id: str = Field(min_length=1)
    default_record_type: str = Field(min_length=1)


class WorkingMemorySettings(StrictBaseModel):
    item_capacity: int = Field(default=32, gt=0, le=4096)
    projection_max_bytes: int = Field(default=2048, gt=0, le=16 * 1024 * 1024)


class SleepSettings(StrictBaseModel):
    enabled: bool
    dream_dataset_path: Path
    min_emotion_score: float = Field(ge=0.0, le=1.0)
    max_episodes_per_cycle: int = Field(gt=0)


class QloraSettings(StrictBaseModel):
    output_dir: Path
    dry_run: bool
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    lora_dropout: float = Field(ge=0.0, lt=1.0)
    learning_rate: float = Field(gt=0.0)
    num_train_epochs: int = Field(gt=0)
    max_steps: int = Field(gt=0)


class AdapterRegistrySettings(StrictBaseModel):
    path: Path
    eval_result_dir: Path
    eval_sets: list[Path]
    trial_threshold: float = Field(ge=0.0, le=1.0)
    reject_threshold: float = Field(ge=0.0, le=1.0)
    allowed_states: list[str]
    manual_approval_required: bool


class ApiSettings(StrictBaseModel):
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    admin_token_env: str = Field(min_length=1)
    cors_origins: list[str]


class FrontendSettings(StrictBaseModel):
    base_url: str = Field(min_length=1)
    api_base_url: str = Field(min_length=1)


class RuntimeSettings(StrictBaseModel):
    queue_capacity: int = Field(default=64, gt=0)


class AgentStateSettings(StrictBaseModel):
    path: Path = Path(".kagya/agent_state.json")


class EventJournalSettings(StrictBaseModel):
    path: Path = Path(".kagya/event_journal.jsonl")
    max_bytes: int = Field(default=1_048_576, gt=0)
    retained_files: int = Field(default=4, ge=2)


class StateWALSettings(StrictBaseModel):
    directory: Path = Path(".kagya/private/state_wal")


class ValueSeedSettings(StrictBaseModel):
    """A non-authoritative value declaration supplied by configuration."""

    value_id: StrictStr
    name: StrictStr
    concept: StrictStr | None = None
    scope: Literal["subject", "context"] = "subject"
    context_ids: list[StrictStr] = Field(default_factory=list, max_length=16)
    polarity: StrictInt = 1
    strength: StrictFloat = Field(ge=0.0, le=1.0)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    stability: StrictFloat = Field(ge=0.0, le=1.0)
    protectedness: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    negotiability: StrictFloat = Field(default=1.0, ge=0.0, le=1.0)
    allowed_update_rate: StrictFloat = Field(gt=0.0, le=1.0)

    @field_validator("value_id")
    @classmethod
    def validate_value_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value) or len(value) > 128:
            raise ValueError("value text must be bounded and single-line")
        return value

    @field_validator("concept")
    @classmethod
    def validate_concept(cls, value: str | None) -> str | None:
        if value is not None and (
            not value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or len(value) > 2048
        ):
            raise ValueError("value text must be bounded and single-line")
        return value

    @field_validator("context_ids")
    @classmethod
    def validate_context_ids(cls, value: list[str]) -> list[str]:
        for context_id in value:
            validate_identifier(context_id)
        if value != sorted(set(value)) or len(value) != len(set(value)):
            raise ValueError("context_ids must be sorted and unique")
        return value

    @field_validator(
        "strength",
        "confidence",
        "stability",
        "protectedness",
        "negotiability",
        "allowed_update_rate",
    )
    @classmethod
    def require_finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value settings must be finite")
        return value

    @field_validator("polarity")
    @classmethod
    def validate_polarity(cls, value: int) -> int:
        if value not in (-1, 1):
            raise ValueError("polarity must be exactly -1 or 1")
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> "ValueSeedSettings":
        if self.scope == "subject" and self.context_ids:
            raise ValueError("subject values cannot have contexts")
        if self.scope == "context" and not self.context_ids:
            raise ValueError("context values require contexts")
        return self


class ValueConflictSettings(StrictBaseModel):
    conflict_id: StrictStr
    name: StrictStr
    left_value_id: StrictStr
    right_value_id: StrictStr

    @field_validator("conflict_id", "name")
    @classmethod
    def validate_conflict_text(cls, value: str) -> str:
        if not value or len(value) > 128 or any(
            ord(char) < 32 or ord(char) == 127 for char in value
        ):
            raise ValueError("conflict text must be bounded and single-line")
        return value

    @field_validator("conflict_id")
    @classmethod
    def validate_conflict_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("left_value_id", "right_value_id")
    @classmethod
    def validate_conflict_value_id(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def validate_pair_order(self) -> "ValueConflictSettings":
        if self.left_value_id == self.right_value_id:
            raise ValueError("conflicts cannot be self-conflicts")
        if self.left_value_id > self.right_value_id:
            raise ValueError("conflict value IDs must be in canonical order")
        return self


class ValueSystemSettings(StrictBaseModel):
    seeds: list[ValueSeedSettings] = Field(default_factory=list, max_length=128)
    conflicts: list[ValueConflictSettings] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def validate_references(self) -> "ValueSystemSettings":
        seed_ids = [seed.value_id for seed in self.seeds]
        present_ids = set(seed_ids)
        if len(present_ids) != len(seed_ids):
            raise ValueError("seed value IDs must be unique")
        pairs: set[tuple[str, str]] = set()
        conflict_ids: set[str] = set()
        for conflict in self.conflicts:
            pair = (conflict.left_value_id, conflict.right_value_id)
            if not {conflict.left_value_id, conflict.right_value_id} <= present_ids:
                raise ValueError("conflicts must reference known seed value IDs")
            if pair in pairs or (pair[1], pair[0]) in pairs:
                raise ValueError("conflict pairs must be unique")
            if conflict.conflict_id in conflict_ids:
                raise ValueError("conflict IDs must be unique")
            pairs.add(pair)
            conflict_ids.add(conflict.conflict_id)
        return self


class Settings(StrictBaseModel):
    project: ProjectSettings
    model: ModelSettings
    generation: GenerationSettings
    emotion: EmotionSettings
    appraisal: AppraisalSettings = Field(default_factory=AppraisalSettings)
    memory: MemorySettings
    sleep: SleepSettings
    qlora: QloraSettings
    adapter_registry: AdapterRegistrySettings
    api: ApiSettings
    frontend: FrontendSettings
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    working_memory: WorkingMemorySettings = Field(default_factory=WorkingMemorySettings)
    agent_state: AgentStateSettings = Field(default_factory=AgentStateSettings)
    event_journal: EventJournalSettings = Field(default_factory=EventJournalSettings)
    state_wal: StateWALSettings = Field(default_factory=StateWALSettings)
    values: ValueSystemSettings = Field(default_factory=ValueSystemSettings)
