"""Validated local import of remote candidate adapter artifacts."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from kagya.config import Settings
from kagya.learning.adapter_registry import AdapterEntry, AdapterRegistry
from kagya.training.artifacts import TrainingArtifactContract, sha256_file_map


@dataclass(frozen=True)
class AdapterImportAttempt:
    attempt_id: str
    job_id: str
    adapter_id: str | None
    adapter_hash: str | None
    status: str
    created_at: str
    updated_at: str
    error: str | None = None
    schema_version: int = 1


class CandidateArtifactImporter:
    def __init__(
        self,
        settings: Settings,
        registry: AdapterRegistry,
        *,
        load_smoke: Callable[[Path], None] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.contract = TrainingArtifactContract()
        self.import_root = settings.qlora.output_dir / "imported"
        self.attempt_path = settings.adapter_registry.path.with_name(
            f"{settings.adapter_registry.path.stem}_imports.json"
        )
        self._lock_path = self.attempt_path.with_suffix(".lock")
        self.load_smoke = load_smoke or self._default_load_smoke

    def import_result(self, result_path: Path, bundle_path: Path) -> AdapterEntry:
        result = self.contract.validate_result(
            result_path,
            expected_model_id=self.settings.model.primary_id,
            expected_model_revision=self.settings.model.revision,
        )
        bundle = self.contract.validate_bundle(
            bundle_path,
            expected_model_id=self.settings.model.primary_id,
            expected_model_revision=self.settings.model.revision,
            expected_processor_revision=self.settings.model.processor_revision,
            expected_parent_adapter_id=result.parent_adapter_id,
            expected_parent_adapter_hash=result.parent_adapter_hash,
        )
        if result.status != "succeeded" or result.candidate_adapter_id is None:
            raise ValueError("only successful training results can be imported")
        if result.job_id != bundle.job_id or result.attempt_id != bundle.attempt_id:
            raise ValueError("training result and bundle provenance mismatch")
        adapter_files = {
            item.relative_to(result_path).as_posix(): item.read_bytes()
            for item in (result_path / "adapter").rglob("*")
            if item.is_file() and not item.is_symlink()
        }
        adapter_hash = sha256_file_map(adapter_files)
        if adapter_hash != result.candidate_adapter_hash:
            raise ValueError("candidate adapter hash mismatch")
        config_path = result_path / "adapter" / "adapter_config.json"
        config = json.loads(config_path.read_text("utf-8"))
        if not isinstance(config, dict) or not config.get("peft_type"):
            raise ValueError("adapter_config.json is not a PEFT adapter config")
        configured_base = config.get("base_model_name_or_path")
        if configured_base is not None and configured_base != result.base_model_id:
            raise ValueError("adapter config base model mismatch")

        existing = self._existing(result.job_id, adapter_hash)
        if existing is not None:
            return existing
        attempt = self._record(
            result.job_id, result.candidate_adapter_id, adapter_hash, "validating"
        )
        final_path = self.import_root / result.candidate_adapter_id
        staging = self.import_root / f".{result.candidate_adapter_id}.{uuid4()}.tmp"
        try:
            self.load_smoke(result_path / "adapter")
            self.import_root.mkdir(parents=True, exist_ok=True)
            if not final_path.exists():
                shutil.copytree(result_path / "adapter", staging)
                os.rename(staging, final_path)
            self.load_smoke(final_path)
            training_manifest = final_path / "training_manifest.json"
            try:
                entry = self.registry.register_candidate(
                    adapter_id=result.candidate_adapter_id,
                    adapter_path=final_path.resolve(),
                    dataset_path=(bundle_path / bundle.dataset_path).resolve(),
                    dataset_hash=bundle.dataset_hash,
                    base_model=result.base_model_id,
                    base_model_revision=result.base_model_revision,
                    adapter_hash=adapter_hash,
                    parent_adapter_id=result.parent_adapter_id,
                    parent_adapter_hash=result.parent_adapter_hash,
                    training_job_id=result.job_id,
                    training_node_id=result.worker_node_id,
                    submitted_by_node_id=bundle.submitter_node_id,
                    imported_by_node_id=self.settings.deployment.node.id,
                    training_manifest_path=(
                        str(training_manifest.resolve())
                        if training_manifest.exists()
                        else None
                    ),
                    worker_evaluation_path=str(
                        (result_path / result.evaluation_path).resolve()
                    ),
                    evaluation_set_hashes=(bundle.evaluation_set_hash,),
                    evaluation_dataset_path=(
                        bundle_path / bundle.evaluation_set_path
                    ).resolve(),
                    notes="validated local import from TrainingResult",
                )
            except ValueError:
                existing_entry = self._existing(result.job_id, adapter_hash)
                if existing_entry is None:
                    raise
                entry = existing_entry
            self.validate_entry(entry)
            self._update(attempt.attempt_id, "completed")
            return entry
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            self._update(attempt.attempt_id, "failed", str(exc))
            raise

    def _existing(self, job_id: str, adapter_hash: str) -> AdapterEntry | None:
        by_job = next(
            (
                entry
                for entry in self.registry.list()
                if entry.training_job_id == job_id
            ),
            None,
        )
        by_hash = next(
            (
                entry
                for entry in self.registry.list()
                if entry.adapter_hash == adapter_hash
            ),
            None,
        )
        if by_job is None and by_hash is None:
            return None
        if by_job is not None and by_job == by_hash:
            return by_job
        raise ValueError("adapter hash or training job conflicts with registry")

    def validate_entry(self, entry: AdapterEntry) -> None:
        path = Path(entry.path)
        if not path.is_absolute() or not path.is_relative_to(
            self.import_root.resolve()
        ):
            raise ValueError("registry adapter path is not a local imported artifact")
        files = {
            (Path("adapter") / item.relative_to(path)).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file() and not item.is_symlink()
        }
        if entry.adapter_hash is None or sha256_file_map(files) != entry.adapter_hash:
            raise ValueError("registry adapter hash does not match local artifact")

    def _record(
        self, job_id: str, adapter_id: str, adapter_hash: str, status: str
    ) -> AdapterImportAttempt:
        now = _now()
        attempt = AdapterImportAttempt(
            str(uuid4()), job_id, adapter_id, adapter_hash, status, now, now
        )
        with self._locked_attempts() as attempts:
            attempts.append(attempt)
        return attempt

    def _update(self, attempt_id: str, status: str, error: str | None = None) -> None:
        with self._locked_attempts() as attempts:
            for index, attempt in enumerate(attempts):
                if attempt.attempt_id == attempt_id:
                    attempts[index] = AdapterImportAttempt(
                        **{
                            **asdict(attempt),
                            "status": status,
                            "updated_at": _now(),
                            "error": error,
                        }
                    )
                    return

    def _locked_attempts(self):
        return _AttemptLock(self.attempt_path, self._lock_path)

    @staticmethod
    def _default_load_smoke(path: Path) -> None:
        from peft import PeftConfig

        PeftConfig.from_pretrained(str(path))


class _AttemptLock:
    def __init__(self, path: Path, lock_path: Path) -> None:
        self.path = path
        self.lock_path = lock_path

    def __enter__(self) -> list[AdapterImportAttempt]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.lock_path.open("a+b")
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX)
        if not self.path.exists():
            self.attempts = []
        else:
            raw = json.loads(self.path.read_text("utf-8"))
            self.attempts = [
                AdapterImportAttempt(**item) for item in raw.get("attempts", [])
            ]
        return self.attempts

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            temporary = self.path.with_name(f".{self.path.name}.{uuid4()}.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "attempts": [asdict(item) for item in self.attempts],
                    },
                    sort_keys=True,
                ),
                "utf-8",
            )
            os.replace(temporary, self.path)
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
        self._lock.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()
