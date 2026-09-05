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


class EmotionSettings(StrictBaseModel):
    baseline_surprisal: float = Field(ge=0.0)
    high_emotion_threshold: float = Field(ge=0.0, le=1.0)
    decay_rate: float = Field(ge=0.0)


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


class Settings(StrictBaseModel):
    project: ProjectSettings
    model: ModelSettings
    generation: GenerationSettings
    emotion: EmotionSettings
    memory: MemorySettings
    sleep: SleepSettings
    qlora: QloraSettings
    adapter_registry: AdapterRegistrySettings
    api: ApiSettings
    frontend: FrontendSettings
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    agent_state: AgentStateSettings = Field(default_factory=AgentStateSettings)
