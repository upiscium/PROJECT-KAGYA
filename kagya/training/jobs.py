"""Persistent asynchronous sleep-training job lifecycle."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import json
import os
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, Protocol
from uuid import uuid4

from kagya.config import Settings
from kagya.learning.adapter_registry import AdapterRegistry
from kagya.learning.dream_dataset_generator import DreamDatasetRecord
from kagya.learning.qlora_trainer import QloraTrainer, QloraTrainingResult
from kagya.memory import (
    ConsolidationStatus,
    DualMemorySystem,
    EpisodicMemoryRecord,
    ValidationStatus,
)
from kagya.models import ModelProvider
from kagya.training.artifacts import (
    TrainingArtifactContract,
    TrainingBundleManifest,
    sha256_bytes,
)


class TrainingJobStatus(StrEnum):
    PREPARING = "preparing"
    READY = "ready"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    IMPORTING = "importing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES = {
    TrainingJobStatus.COMPLETED,
    TrainingJobStatus.FAILED,
    TrainingJobStatus.CANCELLED,
}


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    attempt_id: str
    idempotency_key: str
    status: TrainingJobStatus
    bundle_path: str | None
    bundle_hash: str | None
    base_model_id: str
    base_model_revision: str
    parent_adapter_id: str | None
    source_event_sequence_start: int
    source_event_sequence_end: int
    backend: str
    remote_job_id: str | None
    candidate_adapter_id: str | None
    selected_episode_ids: tuple[str, ...]
    semantic_memory_ids: tuple[str, ...]
    created_at: str
    updated_at: str
    error: str | None = None
    retry_count: int = 0
    schema_version: int = 1

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "TrainingJob":
        data = dict(value)
        data["status"] = TrainingJobStatus(data["status"])
        data["selected_episode_ids"] = tuple(data.get("selected_episode_ids", ()))
        data["semantic_memory_ids"] = tuple(data.get("semantic_memory_ids", ()))
        return cls(**data)


class TrainingJobRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._jobs: dict[str, TrainingJob] = {}
        self._load()

    def create(
        self,
        *,
        idempotency_key: str,
        base_model_id: str,
        base_model_revision: str,
        parent_adapter_id: str | None,
        backend: str,
        job_id: str | None = None,
    ) -> tuple[TrainingJob, bool]:
        with self._lock:
            existing = next(
                (
                    job
                    for job in self._jobs.values()
                    if job.idempotency_key == idempotency_key
                ),
                None,
            )
            if existing is not None:
                return existing, False
            now = _now()
            identifier = job_id or str(uuid4())
            job = TrainingJob(
                job_id=identifier,
                attempt_id=str(uuid4()),
                idempotency_key=idempotency_key,
                status=TrainingJobStatus.PREPARING,
                bundle_path=None,
                bundle_hash=None,
                base_model_id=base_model_id,
                base_model_revision=base_model_revision,
                parent_adapter_id=parent_adapter_id,
                source_event_sequence_start=0,
                source_event_sequence_end=0,
                backend=backend,
                remote_job_id=None,
                candidate_adapter_id=None,
                selected_episode_ids=(),
                semantic_memory_ids=(),
                created_at=now,
                updated_at=now,
            )
            self._jobs[identifier] = job
            self._save()
            return job, True

    def get(self, job_id: str) -> TrainingJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError(f"Unknown training job: {job_id}")
            return job

    def list(self) -> list[TrainingJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at)

    def update(self, job_id: str, **changes: Any) -> TrainingJob:
        with self._lock:
            current = self.get(job_id)
            updated = replace(current, updated_at=_now(), **changes)
            self._jobs[job_id] = updated
            self._save()
            return updated

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text("utf-8"))
        jobs = raw.get("jobs", []) if isinstance(raw, dict) else []
        self._jobs = {
            job.job_id: job
            for item in jobs
            if isinstance(item, dict)
            for job in [TrainingJob.from_json(item)]
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4()}.tmp")
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(
                {"schema_version": 1, "jobs": [asdict(job) for job in self.list()]},
                output,
                ensure_ascii=False,
                sort_keys=True,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)
        _fsync_directory(self.path.parent)


class TrainingBackend(Protocol):
    def submit(self, job: TrainingJob, bundle_path: Path) -> str: ...
    def inspect(self, job_id: str) -> TrainingJobStatus: ...
    def cancel(self, job_id: str) -> bool: ...
    def fetch_result(self, job_id: str) -> QloraTrainingResult | None: ...
    def attach(self, job: TrainingJob) -> None: ...
    def shutdown(self) -> None: ...


class LocalTrainingBackend:
    def __init__(self, trainer: QloraTrainer) -> None:
        self.trainer = trainer
        self._statuses: dict[str, TrainingJobStatus] = {}
        self._results: dict[str, QloraTrainingResult] = {}
        self._cancelled: set[str] = set()

    def submit(self, job: TrainingJob, bundle_path: Path) -> str:
        if job.job_id in self._statuses:
            return job.job_id
        self._statuses[job.job_id] = TrainingJobStatus.RUNNING
        if job.job_id in self._cancelled:
            self._statuses[job.job_id] = TrainingJobStatus.CANCELLED
            return job.job_id
        result = self.trainer.train(bundle_path / "dataset.jsonl")
        self._results[job.job_id] = result
        self._statuses[job.job_id] = TrainingJobStatus.SUCCEEDED
        return job.job_id

    def inspect(self, job_id: str) -> TrainingJobStatus:
        return self._statuses[job_id]

    def cancel(self, job_id: str) -> bool:
        if self._statuses.get(job_id) == TrainingJobStatus.RUNNING:
            return False
        self._cancelled.add(job_id)
        self._statuses[job_id] = TrainingJobStatus.CANCELLED
        return True

    def fetch_result(self, job_id: str) -> QloraTrainingResult | None:
        return self._results.get(job_id)

    def attach(self, job: TrainingJob) -> None:
        return None

    def shutdown(self) -> None:
        return None


@dataclass(frozen=True)
class ConsolidationPreparation:
    episodes: tuple[EpisodicMemoryRecord, ...]
    semantic_ids: tuple[str, ...]


class MemoryConsolidator:
    PIPELINE_VERSION = "sleep-job-v1"

    def __init__(
        self, settings: Settings, memory: DualMemorySystem, provider: ModelProvider
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.provider = provider

    def prepare(self, attempt_id: str) -> ConsolidationPreparation:
        episodes = tuple(self._select())
        for episode in episodes:
            self.memory.set_consolidation_state(
                episode.id,
                status=ConsolidationStatus.IN_PROGRESS,
                pipeline_version=self.PIPELINE_VERSION,
                attempt_id=attempt_id,
            )
        semantic_ids = tuple(
            self.memory.save_semantic(
                self.provider.generate(
                    "Extract one concise semantic memory from this high-emotion episode.\n"
                    f"User: {episode.user_input}\nAssistant: {episode.response}"
                ),
                source_episode_ids=[episode.id],
                metadata={
                    "source": "sleep_job",
                    "publication_status": "staged",
                    "attempt_id": attempt_id,
                    "pipeline_version": self.PIPELINE_VERSION,
                    "context_id": episode.context_id,
                },
            )
            for episode in episodes
        )
        return ConsolidationPreparation(episodes, semantic_ids)

    def complete(self, preparation: ConsolidationPreparation, attempt_id: str) -> None:
        for semantic_id in preparation.semantic_ids:
            self.memory.publish_semantic(semantic_id)
        for episode in preparation.episodes:
            self.memory.set_consolidation_state(
                episode.id,
                status=ConsolidationStatus.COMPLETED,
                pipeline_version=self.PIPELINE_VERSION,
                attempt_id=attempt_id,
            )

    def fail(self, preparation: ConsolidationPreparation, attempt_id: str) -> None:
        for semantic_id in preparation.semantic_ids:
            self.memory.archive_semantic(semantic_id)
        for episode in preparation.episodes:
            self.memory.set_consolidation_state(
                episode.id,
                status=ConsolidationStatus.FAILED,
                pipeline_version=self.PIPELINE_VERSION,
                attempt_id=attempt_id,
            )

    def _select(self) -> list[EpisodicMemoryRecord]:
        threshold = self.settings.sleep.min_emotion_score
        return [
            episode
            for episode in self.memory._get_unarchived_episodic_records()
            if (
                episode.emotion_arousal > threshold
                or abs(episode.emotion_valence) > threshold
            )
            and episode.validation_status == ValidationStatus.VERIFIED
            and episode.generation_health.healthy
            and not (
                episode.consolidation_status == ConsolidationStatus.COMPLETED
                and episode.consolidation_version == self.PIPELINE_VERSION
            )
        ][: self.settings.sleep.max_episodes_per_cycle]


class TrainingBundleBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.contract = TrainingArtifactContract()

    def build(self, job: TrainingJob, episodes: tuple[EpisodicMemoryRecord, ...]) -> Path:
        dataset = b"".join(
            (
                json.dumps(
                    DreamDatasetRecord(
                        input=episode.user_input,
                        thought="",
                        output=episode.response,
                        source_id=episode.id,
                        validation_status=episode.validation_status.value,
                    ).to_json(),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode()
            for episode in episodes
        )
        evaluation = b""
        sequences = [
            episode.processing_sequence
            for episode in episodes
            if episode.processing_sequence is not None
        ]
        manifest = TrainingBundleManifest(
            job_id=job.job_id,
            attempt_id=job.attempt_id,
            created_at=datetime.now(UTC),
            submitter_node_id=self.settings.deployment.node.id,
            submitter_hostname=os.uname().nodename,
            base_model_id=self.settings.model.primary_id,
            base_model_revision=self.settings.model.revision,
            processor_revision=self.settings.model.processor_revision,
            parent_adapter_id=job.parent_adapter_id,
            parent_adapter_hash=None,
            source_event_sequence_start=min(sequences, default=0),
            source_event_sequence_end=max(sequences, default=0),
            source_episode_ids=[episode.id for episode in episodes],
            source_decision_ids=[],
            dataset_hash=sha256_bytes(dataset),
            dataset_record_count=len(episodes),
            evaluation_set_hash=sha256_bytes(evaluation),
            evaluation_record_count=0,
            chat_template_version="gemma-v1",
            dataset_format_version="dream-v2",
            qlora_hyperparameters={
                "r": self.settings.qlora.r,
                "alpha": self.settings.qlora.lora_alpha,
                "dropout": self.settings.qlora.lora_dropout,
                "learning_rate": self.settings.qlora.learning_rate,
                "max_steps": self.settings.qlora.max_steps,
            },
            required_capabilities=["qlora"],
        )
        return self.contract.finalize_bundle(
            self.settings.sleep.training_artifact_directory,
            manifest,
            dataset=dataset,
            evaluation_set=evaluation,
        )


class SleepCoordinator:
    def __init__(
        self,
        settings: Settings,
        consolidator: MemoryConsolidator,
        bundle_builder: TrainingBundleBuilder,
        registry: TrainingJobRegistry,
        backend: TrainingBackend,
        adapter_registry: AdapterRegistry,
        subject_executor: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.consolidator = consolidator
        self.bundle_builder = bundle_builder
        self.registry = registry
        self.backend = backend
        self.adapter_registry = adapter_registry
        self.subject_executor = subject_executor or (lambda _source, handler: handler())
        self._threads: dict[str, Thread] = {}
        self._cancel: dict[str, Event] = {}
        for job in self.registry.list():
            if job.backend == "local" and job.status not in TERMINAL_JOB_STATUSES:
                self.registry.update(
                    job.job_id,
                    status=TrainingJobStatus.FAILED,
                    error="local runtime restarted before job completion",
                )
            elif job.backend != "local" and job.status not in TERMINAL_JOB_STATUSES:
                self.backend.attach(job)
                self._start_resume(job)

    def create_job(self, idempotency_key: str) -> TrainingJob:
        job, created = self.registry.create(
            idempotency_key=idempotency_key,
            base_model_id=self.settings.model.primary_id,
            base_model_revision=self.settings.model.revision,
            parent_adapter_id=None,
            backend=self.settings.deployment.training.backend.value,
        )
        if not created:
            return job
        self._start(job)
        return job

    def retry(self, job_id: str) -> TrainingJob:
        previous = self.registry.get(job_id)
        if previous.status not in {
            TrainingJobStatus.FAILED,
            TrainingJobStatus.CANCELLED,
        }:
            raise ValueError("Only failed or cancelled jobs can be retried")
        job, _ = self.registry.create(
            idempotency_key=f"retry:{previous.job_id}:{previous.retry_count + 1}",
            base_model_id=previous.base_model_id,
            base_model_revision=previous.base_model_revision,
            parent_adapter_id=previous.parent_adapter_id,
            backend=previous.backend,
        )
        job = self.registry.update(job.job_id, retry_count=previous.retry_count + 1)
        self._start(job)
        return job

    def _start(self, job: TrainingJob) -> None:
        cancel = Event()
        self._cancel[job.job_id] = cancel
        thread = Thread(
            target=self._run,
            args=(job.job_id, cancel),
            name=f"kagya-sleep-{job.job_id}",
            daemon=True,
        )
        self._threads[job.job_id] = thread
        thread.start()

    def _start_resume(self, job: TrainingJob) -> None:
        thread = Thread(
            target=self._resume,
            args=(job.job_id,),
            name=f"kagya-sleep-resume-{job.job_id}",
            daemon=True,
        )
        self._threads[job.job_id] = thread
        thread.start()

    def inspect(self, job_id: str) -> TrainingJob:
        return self.registry.get(job_id)

    def list_jobs(self) -> list[TrainingJob]:
        return self.registry.list()

    def cancel(self, job_id: str) -> TrainingJob:
        job = self.registry.get(job_id)
        if job.status in TERMINAL_JOB_STATUSES:
            return job
        self._cancel.setdefault(job_id, Event()).set()
        self.backend.cancel(job_id)
        return self.registry.update(job_id, status=TrainingJobStatus.CANCELLED)

    def shutdown(self) -> None:
        shutdown = getattr(self.backend, "shutdown", None)
        if shutdown is not None:
            shutdown()
        for thread in tuple(self._threads.values()):
            thread.join()

    def _resume(self, job_id: str) -> None:
        job = self.registry.get(job_id)
        episodes = tuple(
            episode
            for episode_id in job.selected_episode_ids
            if (episode := self.consolidator.memory.get_episodic(episode_id))
            is not None
        )
        preparation = ConsolidationPreparation(episodes, job.semantic_memory_ids)
        try:
            if job.bundle_path is None:
                raise RuntimeError("remote training job has no persisted bundle")
            remote_id = self.backend.submit(job, Path(job.bundle_path))
            self.registry.update(job_id, remote_job_id=remote_id)
            result = self.backend.fetch_result(job_id)
            if result is None:
                raise RuntimeError("Training backend returned no result")
            self.registry.update(job_id, status=TrainingJobStatus.IMPORTING)
            entry = self.subject_executor(
                "runtime.sleep.resume_finalize",
                lambda: self._finalize_subject_state(
                    preparation, job.attempt_id, result
                ),
            )
            self.registry.update(
                job_id,
                status=TrainingJobStatus.COMPLETED,
                candidate_adapter_id=entry.adapter_id,
                error=None,
            )
        except InterruptedError:
            return
        except Exception as exc:
            self.subject_executor(
                "runtime.sleep.resume_fail",
                lambda: self.consolidator.fail(preparation, job.attempt_id),
            )
            self.registry.update(
                job_id, status=TrainingJobStatus.FAILED, error=str(exc)
            )

    def _run(self, job_id: str, cancel: Event) -> None:
        preparation = ConsolidationPreparation((), ())
        job = self.registry.get(job_id)
        try:
            preparation = self.subject_executor(
                "runtime.sleep.prepare",
                lambda: self.consolidator.prepare(job.attempt_id),
            )
            if not preparation.episodes:
                self.registry.update(job_id, status=TrainingJobStatus.COMPLETED)
                return
            if cancel.is_set():
                raise _Cancelled
            bundle = self.bundle_builder.build(job, preparation.episodes)
            bundle_hash = sha256_bytes((bundle / "checksums.sha256").read_bytes())
            sequences = [
                episode.processing_sequence
                for episode in preparation.episodes
                if episode.processing_sequence is not None
            ]
            job = self.registry.update(
                job_id,
                status=TrainingJobStatus.READY,
                bundle_path=str(bundle),
                bundle_hash=bundle_hash,
                selected_episode_ids=tuple(item.id for item in preparation.episodes),
                semantic_memory_ids=preparation.semantic_ids,
                source_event_sequence_start=min(sequences, default=0),
                source_event_sequence_end=max(sequences, default=0),
            )
            if cancel.is_set():
                raise _Cancelled
            self.registry.update(job_id, status=TrainingJobStatus.DISPATCHED)
            self.registry.update(job_id, status=TrainingJobStatus.RUNNING)
            remote_id = self.backend.submit(job, bundle)
            self.registry.update(job_id, remote_job_id=remote_id)
            if cancel.is_set():
                raise _Cancelled
            result = self.backend.fetch_result(job_id)
            if result is None:
                raise RuntimeError("Training backend returned no result")
            self.registry.update(
                job_id,
                status=TrainingJobStatus.SUCCEEDED,
            )
            self.registry.update(job_id, status=TrainingJobStatus.IMPORTING)
            entry = self.subject_executor(
                "runtime.sleep.finalize",
                lambda: self._finalize_subject_state(
                    preparation, job.attempt_id, result
                ),
            )
            self.registry.update(
                job_id,
                status=TrainingJobStatus.COMPLETED,
                candidate_adapter_id=entry.adapter_id,
            )
        except _Cancelled:
            self.subject_executor(
                "runtime.sleep.cancel",
                lambda: self.consolidator.fail(preparation, job.attempt_id),
            )
            self.registry.update(job_id, status=TrainingJobStatus.CANCELLED)
        except Exception as exc:
            self.subject_executor(
                "runtime.sleep.fail",
                lambda: self.consolidator.fail(preparation, job.attempt_id),
            )
            self.registry.update(
                job_id, status=TrainingJobStatus.FAILED, error=str(exc)
            )

    def _finalize_subject_state(
        self,
        preparation: ConsolidationPreparation,
        attempt_id: str,
        result: QloraTrainingResult,
    ):
        entry = self.adapter_registry.lookup(result.adapter_id)
        if entry is None:
            entry = self.adapter_registry.register_candidate(
                adapter_id=result.adapter_id,
                adapter_path=result.adapter_path,
                dataset_path=result.dataset_path,
                dataset_hash=result.dataset_hash,
                base_model=self.settings.model.primary_id,
                notes=f"registered by {self.settings.deployment.training.backend.value} sleep job",
            )
        self.consolidator.complete(preparation, attempt_id)
        return entry


class _Cancelled(Exception):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
