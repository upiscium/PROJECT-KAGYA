"""Operator-invoked end-to-end smoke test for an SSH training worker."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import socket
from typing import Any
from uuid import uuid4

from kagya.config import Settings, load_settings
from kagya.learning import (
    AdapterRegistry,
    AdapterRuntimeManager,
    RuntimeAdapterState,
)
from kagya.models import DummyProvider
from kagya.runtime import AgentEventType, AgentRuntime
from kagya.training.artifacts import (
    TrainingArtifactContract,
    TrainingBundleManifest,
    sha256_bytes,
)
from kagya.training.importer import CandidateArtifactImporter
from kagya.training.jobs import TrainingJob, TrainingJobStatus
from kagya.training.remote import SSHTrainingBackend


CONFIRMATION = "RUN-SPLIT-TRAINING"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"Refusing to train without --confirm {CONFIRMATION}")
    result = run_split_smoke(load_settings(), args.work_dir.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


def run_split_smoke(settings: Settings, work_dir: Path) -> dict[str, Any]:
    remote = settings.deployment.training.remote_worker
    if settings.deployment.training.backend.value != "ssh" or remote is None:
        raise ValueError("split smoke requires the SSH training backend")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError("split smoke work directory must be empty")
    work_dir.mkdir(parents=True, exist_ok=True)

    backend = SSHTrainingBackend(remote, work_dir / "transport")
    health = backend.node_status()
    if not health.get("reachable"):
        raise RuntimeError(f"training worker is unreachable: {health.get('error')}")
    if health.get("node_id") != remote.node_id:
        raise RuntimeError("training worker returned an unexpected node ID")
    if not health.get("gpu", {}).get("available"):
        raise RuntimeError("training worker reports no CUDA GPU")

    bundle, job = _bundle_and_job(settings, work_dir)
    first_remote_id = backend.submit(job, bundle)
    duplicate_remote_id = backend.submit(job, bundle)
    if first_remote_id != duplicate_remote_id:
        raise RuntimeError("duplicate submit created a different remote job")
    training_result = backend.fetch_result(job.job_id)
    if training_result is None or training_result.artifact_path is None:
        raise RuntimeError("remote worker returned no training result")

    isolated = _isolated_settings(settings, work_dir)
    registry = AdapterRegistry(isolated)
    entry = CandidateArtifactImporter(isolated, registry).import_result(
        training_result.artifact_path, bundle
    )
    registry.apply_evaluation(
        entry.adapter_id,
        score=1.0,
        result_path=training_result.artifact_path / "evaluation.json",
    )
    approved = registry.approve(entry.adapter_id, notes="split deployment smoke")
    activation, rollback = _exercise_lifecycle(registry, approved.adapter_id, work_dir)
    restored = registry.lookup(approved.adapter_id)
    if restored is None:
        raise RuntimeError("imported adapter disappeared after rollback")

    return {
        "job_id": job.job_id,
        "remote_job_id": first_remote_id,
        "duplicate_submit_idempotent": True,
        "worker_node_id": health.get("node_id"),
        "worker_hostname": health.get("hostname"),
        "gpu": health.get("gpu"),
        "result_path": str(training_result.artifact_path),
        "adapter_id": restored.adapter_id,
        "adapter_hash": restored.adapter_hash,
        "adapter_status_after_rollback": restored.status.value,
        "activation_sequence": activation.activation_sequence,
        "rollback_sequence": rollback.activation_sequence,
        "registry_path": str(isolated.adapter_registry.path),
    }


def _bundle_and_job(settings: Settings, work_dir: Path) -> tuple[Path, TrainingJob]:
    job_id = f"smoke-{uuid4()}"
    attempt_id = str(uuid4())
    dataset = (
        json.dumps(
            {
                "input": "State one safe operational principle.",
                "thought": "Prefer reversible, observable changes.",
                "output": "Make small changes, verify them, and keep rollback available.",
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    evaluation = b""
    manifest = TrainingBundleManifest(
        job_id=job_id,
        attempt_id=attempt_id,
        created_at=datetime.now(UTC),
        submitter_node_id=settings.deployment.node.id,
        submitter_hostname=socket.gethostname(),
        base_model_id=settings.model.primary_id,
        base_model_revision=settings.model.revision,
        processor_revision=settings.model.processor_revision,
        source_event_sequence_start=0,
        source_event_sequence_end=0,
        dataset_hash=sha256_bytes(dataset),
        dataset_record_count=1,
        evaluation_set_hash=sha256_bytes(evaluation),
        evaluation_record_count=0,
        chat_template_version="gemma-v1",
        dataset_format_version="dream-v2",
        qlora_hyperparameters={
            "r": settings.qlora.r,
            "learning_rate": settings.qlora.learning_rate,
            "max_steps": settings.qlora.max_steps,
            "seed": settings.qlora.seed,
        },
        required_capabilities=["cuda", "bitsandbytes"],
    )
    bundle = TrainingArtifactContract().finalize_bundle(
        work_dir / "bundles", manifest, dataset=dataset, evaluation_set=evaluation
    )
    now = datetime.now(UTC).isoformat()
    job = TrainingJob(
        job_id=job_id,
        attempt_id=attempt_id,
        idempotency_key=f"split-smoke-{job_id}",
        status=TrainingJobStatus.READY,
        bundle_path=str(bundle),
        bundle_hash=sha256_bytes((bundle / "checksums.sha256").read_bytes()),
        base_model_id=settings.model.primary_id,
        base_model_revision=settings.model.revision,
        parent_adapter_id=None,
        source_event_sequence_start=0,
        source_event_sequence_end=0,
        backend="ssh",
        remote_job_id=None,
        candidate_adapter_id=None,
        selected_episode_ids=(),
        semantic_memory_ids=(),
        created_at=now,
        updated_at=now,
        processor_revision=settings.model.processor_revision,
    )
    return bundle, job


def _isolated_settings(settings: Settings, work_dir: Path) -> Settings:
    return settings.model_copy(
        update={
            "qlora": settings.qlora.model_copy(
                update={"output_dir": work_dir / "local-adapters"}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": work_dir / "adapter_registry.json",
                    "eval_result_dir": work_dir / "eval-results",
                    "eval_sets": [],
                }
            ),
        }
    )


def _exercise_lifecycle(
    registry: AdapterRegistry, adapter_id: str, work_dir: Path
) -> tuple[Any, Any]:
    state = [RuntimeAdapterState(None, None, None, DummyProvider())]

    def switch(provider, entry, sequence) -> None:
        state[0] = RuntimeAdapterState(
            None if entry is None else entry.adapter_id,
            None if entry is None else entry.adapter_hash,
            sequence,
            provider,
        )

    manager = AdapterRuntimeManager(
        registry,
        provider_loader=lambda _entry: DummyProvider(),
        runtime_switch=switch,
        runtime_snapshot=lambda: state[0],
        history_path=work_dir / "adapter_registry_activations.json",
    )
    manager.stage(adapter_id)
    manager.verify(adapter_id)
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    try:
        activated = runtime.execute(
            AgentEventType.ADAPTER_UPDATE,
            source="split-smoke.activate",
            handler=lambda: manager.activate_at_event_boundary(adapter_id),
        ).value
        rolled_back = runtime.execute(
            AgentEventType.ADAPTER_UPDATE,
            source="split-smoke.rollback",
            handler=manager.rollback,
        ).value
    finally:
        runtime.shutdown()
    return activated, rolled_back


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kagya.training.split_smoke")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
