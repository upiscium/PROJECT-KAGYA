"""Typed configuration schema for PROJECT-KAGYA."""

from pathlib import Path
from enum import StrEnum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    revision: str = Field(min_length=1)
    processor_revision: str = Field(min_length=1)
    fallback_revision: str = Field(min_length=1)


class DeploymentMode(StrEnum):
    STANDALONE = "standalone"
    SPLIT = "split"


class NodeRole(StrEnum):
    ALL = "all"
    INFERENCE = "inference"
    TRAINING_WORKER = "training_worker"


class TrainingBackendType(StrEnum):
    LOCAL = "local"
    SSH = "ssh"
    WORKER = "worker"


class NodeSettings(StrictBaseModel):
    id: str = Field(min_length=1)
    role: NodeRole
    expected_hostname: str | None = Field(default=None, min_length=1)
    enforce_hostname_match: bool = False

    @model_validator(mode="after")
    def validate_node_id(self) -> "NodeSettings":
        if re.fullmatch(r"[A-Za-z0-9._-]+", self.id) is None:
            raise ValueError("node.id contains unsafe characters")
        if self.enforce_hostname_match and self.expected_hostname is None:
            raise ValueError(
                "expected_hostname is required when hostname matching is enforced"
            )
        return self


class ExpectedWorkerModelSettings(StrictBaseModel):
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    processor_revision: str = Field(min_length=1)


class RemoteWorkerSettings(StrictBaseModel):
    node_id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(default=22, gt=0, le=65535)
    user: str = Field(min_length=1)
    identity_file: Path
    known_hosts_file: Path
    remote_inbox: Path
    remote_results: Path
    command: Path
    connect_timeout_seconds: float = Field(default=10.0, gt=0.0)
    job_timeout_seconds: float = Field(default=86400.0, gt=0.0)
    poll_interval_seconds: float = Field(default=30.0, gt=0.0)
    expected_worker_model: ExpectedWorkerModelSettings
    worker_token_env: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_transport(self) -> "RemoteWorkerSettings":
        if re.fullmatch(r"[A-Za-z0-9._-]+", self.node_id) is None:
            raise ValueError("remote worker node_id contains unsafe characters")
        if re.fullmatch(r"[A-Za-z0-9._-]+", self.host) is None:
            raise ValueError("remote worker host contains unsafe characters")
        if re.fullmatch(r"[A-Za-z0-9._-]+", self.user) is None:
            raise ValueError("remote worker user contains unsafe characters")
        for label, path in (
            ("remote_inbox", self.remote_inbox),
            ("remote_results", self.remote_results),
            ("command", self.command),
        ):
            if (
                not path.is_absolute()
                or re.fullmatch(r"/[A-Za-z0-9._/-]+", str(path)) is None
            ):
                raise ValueError(f"remote worker {label} must be a safe absolute path")
        return self


class WorkerSettings(StrictBaseModel):
    inbox_directory: Path
    work_directory: Path
    result_directory: Path
    max_concurrent_jobs: int = Field(default=1, gt=0)
    retain_failed_jobs: bool = True
    allowed_submitters: list[str] = Field(min_length=1)
    worker_token_env: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_submitters(self) -> "WorkerSettings":
        if len(self.allowed_submitters) != len(set(self.allowed_submitters)):
            raise ValueError("allowed submitter node IDs must be unique")
        if any(
            re.fullmatch(r"[A-Za-z0-9._-]+", item) is None
            for item in self.allowed_submitters
        ):
            raise ValueError("allowed submitter node ID contains unsafe characters")
        directories = {
            self.inbox_directory.resolve(),
            self.work_directory.resolve(),
            self.result_directory.resolve(),
        }
        if len(directories) != 3:
            raise ValueError(
                "worker inbox, work, and result directories must be distinct"
            )
        return self


class DeploymentTrainingSettings(StrictBaseModel):
    backend: TrainingBackendType
    remote_worker: RemoteWorkerSettings | None = None
    worker: WorkerSettings | None = None


class DeploymentSettings(StrictBaseModel):
    mode: DeploymentMode
    node: NodeSettings
    training: DeploymentTrainingSettings


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
    job_registry_path: Path = Path(".kagya/training_jobs.json")
    training_artifact_directory: Path = Path(".kagya/training_artifacts")
    artifact_retention_days: int = Field(default=30, ge=1)


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
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    max_sequence_length: int = Field(default=512, gt=0)
    optimizer: Literal["paged_adamw_8bit"] = "paged_adamw_8bit"
    seed: int = Field(default=42, ge=0)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        min_length=1,
    )
    resume_policy: Literal["never"] = "never"

    @model_validator(mode="after")
    def validate_target_modules(self) -> "QloraSettings":
        if len(self.target_modules) != len(set(self.target_modules)):
            raise ValueError("qlora.target_modules must be unique")
        if any(
            re.fullmatch(r"[A-Za-z0-9_]+", item) is None for item in self.target_modules
        ):
            raise ValueError("qlora.target_modules contains an unsafe module name")
        return self


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


class AgentJournalSettings(StrictBaseModel):
    path: Path = Path(".kagya/agent_journal.jsonl")
    max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    retained_files: int = Field(default=3, ge=1, le=100)


class ObservabilitySettings(StrictBaseModel):
    enabled: bool = True
    metrics_path: Path = Path(".kagya/operational_metrics.json")
    traces_path: Path = Path(".kagya/operational_traces.json")
    max_series: int = Field(default=512, ge=32, le=10000)
    max_traces: int = Field(default=1000, ge=10, le=100000)


class WorkingMemorySettings(StrictBaseModel):
    item_capacity: int = Field(default=32, gt=0)
    token_capacity: int = Field(default=2048, gt=0)


class ValueSeedSettings(StrictBaseModel):
    value_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    source: str = Field(default="config", min_length=1)
    origin: str = Field(default="persona:default", min_length=1)
    allowed_update_rate: float = Field(default=0.05, gt=0.0, le=1.0)


class ValueConflictSettings(StrictBaseModel):
    left_value_id: str = Field(min_length=1)
    right_value_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class ValueSystemSettings(StrictBaseModel):
    max_update_per_event: float = Field(default=0.05, gt=0.0, le=1.0)
    max_total_update_per_event: float = Field(default=0.1, gt=0.0, le=1.0)
    seeds: list[ValueSeedSettings] = Field(default_factory=list)
    conflicts: list[ValueConflictSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "ValueSystemSettings":
        value_ids = [seed.value_id for seed in self.seeds]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("value seed IDs must be unique")
        known = set(value_ids)
        for conflict in self.conflicts:
            if (
                conflict.left_value_id not in known
                or conflict.right_value_id not in known
            ):
                raise ValueError("value conflict references unknown seed")
            if conflict.left_value_id == conflict.right_value_id:
                raise ValueError("value conflict must reference two values")
        return self


class ApiSettings(StrictBaseModel):
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    admin_token_env: str = Field(min_length=1)
    agent_queue_capacity: int = Field(default=32, gt=0)
    cors_origins: list[str]
    admin_auth: "AdminAuthSettings" = Field(default_factory=lambda: AdminAuthSettings())


class AdminAuthSettings(StrictBaseModel):
    enabled: bool = False
    actor_header: str = Field(default="X-KAGYA-Actor", min_length=1)
    role_header: str = Field(default="X-KAGYA-Role", min_length=1)
    reauthenticated_at_header: str = Field(
        default="X-KAGYA-Reauthenticated-At", min_length=1
    )
    session_cookie_name: str = Field(default="kagya_admin_session", min_length=1)
    csrf_cookie_name: str = Field(default="kagya_admin_csrf", min_length=1)
    csrf_header: str = Field(default="X-KAGYA-CSRF-Token", min_length=1)
    reauthentication_max_age_seconds: int = Field(default=300, gt=0)
    reauthentication_paths: list[str] = Field(default_factory=list)
    allow_loopback_recovery: bool = True


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
    agent_journal: AgentJournalSettings = Field(default_factory=AgentJournalSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    working_memory: WorkingMemorySettings = Field(default_factory=WorkingMemorySettings)
    values: ValueSystemSettings = Field(default_factory=ValueSystemSettings)
    api: ApiSettings
    frontend: FrontendSettings
    deployment: DeploymentSettings

    @model_validator(mode="after")
    def validate_deployment_topology(self) -> "Settings":
        mode = self.deployment.mode
        role = self.deployment.node.role
        training = self.deployment.training
        combination = (mode, role, training.backend)
        allowed = {
            (
                DeploymentMode.STANDALONE,
                NodeRole.ALL,
                TrainingBackendType.LOCAL,
            ),
            (DeploymentMode.SPLIT, NodeRole.INFERENCE, TrainingBackendType.SSH),
            (
                DeploymentMode.SPLIT,
                NodeRole.TRAINING_WORKER,
                TrainingBackendType.WORKER,
            ),
        }
        if combination not in allowed:
            raise ValueError(
                "invalid deployment mode, node role, and training backend combination"
            )
        if role == NodeRole.INFERENCE:
            remote = training.remote_worker
            if remote is None or training.worker is not None:
                raise ValueError(
                    "split inference requires remote_worker and forbids worker settings"
                )
            expected = remote.expected_worker_model
            if (
                expected.model_id != self.model.primary_id
                or expected.revision != self.model.revision
                or expected.processor_revision != self.model.processor_revision
            ):
                raise ValueError(
                    "remote worker model and revisions must match inference model"
                )
            if self.model.revision == "main" or self.model.processor_revision == "main":
                raise ValueError("split inference requires exact immutable revisions")
        elif role == NodeRole.TRAINING_WORKER:
            if training.worker is None or training.remote_worker is not None:
                raise ValueError(
                    "training worker requires worker settings and forbids remote_worker"
                )
            if self.model.revision == "main" or self.model.processor_revision == "main":
                raise ValueError(
                    "split training worker requires exact immutable revisions"
                )
        elif training.remote_worker is not None or training.worker is not None:
            raise ValueError(
                "standalone deployment forbids remote worker and worker settings"
            )
        if self.agent_state.path.resolve() == self.agent_journal.path.resolve():
            raise ValueError("agent state and journal paths must be distinct")
        operational_paths = {
            self.agent_state.path.resolve(),
            self.agent_journal.path.resolve(),
            self.observability.metrics_path.resolve(),
            self.observability.traces_path.resolve(),
        }
        if len(operational_paths) != 4:
            raise ValueError("state, journal, metrics, and traces paths must be distinct")
        return self
