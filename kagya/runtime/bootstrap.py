"""Role-specific runtime shells for deployment bootstrap."""

from dataclasses import dataclass
from pathlib import Path

from kagya.config.schema import RemoteWorkerSettings, WorkerSettings


@dataclass(frozen=True)
class RemoteTrainingDispatcher:
    """Connection metadata only; transport is implemented by Issue #97."""

    worker_node_id: str
    host: str
    port: int
    user: str
    identity_file: Path
    known_hosts_file: Path
    remote_inbox: Path
    remote_results: Path
    command: Path

    @classmethod
    def from_settings(
        cls, settings: RemoteWorkerSettings
    ) -> "RemoteTrainingDispatcher":
        return cls(
            worker_node_id=settings.node_id,
            host=settings.host,
            port=settings.port,
            user=settings.user,
            identity_file=settings.identity_file,
            known_hosts_file=settings.known_hosts_file,
            remote_inbox=settings.remote_inbox,
            remote_results=settings.remote_results,
            command=settings.command,
        )


@dataclass(frozen=True)
class TrainingWorkerRuntime:
    """Filesystem job runtime that does not initialize subject components."""

    node_id: str
    inbox_directory: Path
    work_directory: Path
    result_directory: Path
    max_concurrent_jobs: int
    retain_failed_jobs: bool
    allowed_submitters: tuple[str, ...]

    @classmethod
    def from_settings(
        cls, node_id: str, settings: WorkerSettings
    ) -> "TrainingWorkerRuntime":
        return cls(
            node_id=node_id,
            inbox_directory=settings.inbox_directory,
            work_directory=settings.work_directory,
            result_directory=settings.result_directory,
            max_concurrent_jobs=settings.max_concurrent_jobs,
            retain_failed_jobs=settings.retain_failed_jobs,
            allowed_submitters=tuple(settings.allowed_submitters),
        )
