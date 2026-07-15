"""Typed configuration schema for PROJECT-KAGYA."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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
    repetition_penalty: float = Field(default=1.1, ge=1.0)
    no_repeat_ngram_size: int = Field(default=6, ge=0)


class EmotionSettings(StrictBaseModel):
    baseline_surprisal: float = Field(ge=0.0)
    high_emotion_threshold: float = Field(ge=0.0, le=1.0)
    decay_rate: float = Field(ge=0.0)
    appraisal_response_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    resting_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    resting_arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    valence_recovery_rate: float = Field(default=0.01, ge=0.0)
    arousal_recovery_rate: float = Field(default=0.02, ge=0.0)


class AppraisalSettings(StrictBaseModel):
    initial_loss_scale: float = Field(default=0.5, gt=0.0)
    minimum_loss_scale: float = Field(default=0.1, gt=0.0)
    default_controllability: float = Field(default=0.6, ge=0.0, le=1.0)
    default_certainty: float = Field(default=0.7, ge=0.0, le=1.0)
    timer_enabled: bool = False
    timer_interval_seconds: float = Field(default=60.0, gt=0.0)


class MemorySettings(StrictBaseModel):
    persist_directory: Path
    db1_collection: str = Field(min_length=1)
    db2_collection: str = Field(min_length=1)
    db1_top_k: int = Field(gt=0)
    db2_top_k: int = Field(gt=0)
    embedding_model_id: str = Field(min_length=1)
    default_record_type: str = Field(min_length=1)


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


class ToolRegistrySettings(StrictBaseModel):
    path: Path
    audit_path: Path


class AgentStateSettings(StrictBaseModel):
    path: Path = Path(".kagya/agent_state.json")


class WorkingMemorySettings(StrictBaseModel):
    item_capacity: int = Field(default=32, gt=0)
    token_capacity: int = Field(default=2048, gt=0)


class ApiSettings(StrictBaseModel):
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    admin_token_env: str = Field(min_length=1)
    agent_queue_capacity: int = Field(default=32, gt=0)
    cors_origins: list[str]


class FrontendSettings(StrictBaseModel):
    base_url: str = Field(min_length=1)
    api_base_url: str = Field(min_length=1)


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
    tools: ToolRegistrySettings
    agent_state: AgentStateSettings = Field(default_factory=AgentStateSettings)
    working_memory: WorkingMemorySettings = Field(default_factory=WorkingMemorySettings)
    api: ApiSettings
    frontend: FrontendSettings
