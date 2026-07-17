from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kagya.api.server import create_app
from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_inference_role_builds_subject_with_admin_remote_sleep(tmp_path: Path) -> None:
    settings = _inference_settings(tmp_path)
    app = create_app(settings)
    app.state.model_provider = DummyProvider()
    app.state.memory_system = DualMemorySystem(
        settings, embedding_function=DeterministicEmbeddingFunction()
    )
    app.state.adapter_registry = AdapterRegistry(settings)

    with TestClient(app) as client:
        assert client.get("/health").json()["role"] == "inference"
        assert client.post(
            "/api/chat", json={"text": "hello", "attachments": []}
        ).status_code == 200
        assert client.post("/api/sleep/jobs", json={}).status_code == 401
        assert app.state.main_loop is not None
        assert app.state.memory_system is not None
        assert app.state.agent_state_store is not None
        assert app.state.training_dispatcher.worker_node_id == "training-01"
        assert not hasattr(app.state, "worker_runtime")
        assert not hasattr(app.state, "sleep_cycle_manager")


def test_training_worker_exposes_health_without_subject_runtime(tmp_path: Path) -> None:
    settings = _worker_settings(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "project": "PROJECT-KAGYA",
            "role": "training_worker",
        }
        assert client.post("/api/chat", json={"text": "hello"}).status_code == 404
        assert client.get("/api/memory/search", params={"query": "x"}).status_code == 404
        assert client.post("/api/sleep/jobs", json={}).status_code == 404
        assert app.state.worker_runtime.node_id == "training-01"
        assert app.state.worker_runtime.max_concurrent_jobs == 1
        for forbidden in (
            "main_loop",
            "memory_system",
            "model_provider",
            "adapter_registry",
            "agent_state_store",
            "agent_runtime",
        ):
            assert not hasattr(app.state, forbidden)


def test_hostname_validation_runs_before_role_bootstrap(tmp_path: Path) -> None:
    settings = _worker_settings(tmp_path)
    raw = settings.model_dump(mode="python")
    raw["deployment"]["node"]["expected_hostname"] = "not-this-host"
    raw["deployment"]["node"]["enforce_hostname_match"] = True
    app = create_app(Settings.model_validate(raw))

    with pytest.raises(RuntimeError, match="startup host"):
        with TestClient(app):
            pass


def _base_settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_role_test",
                    "db2_collection": "cortex_role_test",
                }
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": [],
                }
            ),
            "agent_state": settings.agent_state.model_copy(
                update={"path": tmp_path / "agent_state.json"}
            ),
            "agent_journal": settings.agent_journal.model_copy(
                update={"path": tmp_path / "agent_journal.jsonl"}
            ),
        }
    )


def _inference_settings(tmp_path: Path) -> Settings:
    settings = _base_settings(tmp_path)
    raw = settings.model_dump(mode="python")
    raw["model"]["revision"] = "model-commit-123"
    raw["model"]["processor_revision"] = "processor-commit-123"
    raw["deployment"] = {
        "mode": "split",
        "node": {"id": "inference-01", "role": "inference"},
        "training": {
            "backend": "ssh",
            "remote_worker": {
                "node_id": "training-01",
                "host": "training.local",
                "user": "worker",
                "identity_file": tmp_path / "id_worker",
                "known_hosts_file": tmp_path / "known_hosts",
                "remote_inbox": "/worker/inbox",
                "remote_results": "/worker/results",
                "command": "/worker/bin/kagya-worker",
                "expected_worker_model": {
                    "model_id": raw["model"]["primary_id"],
                    "revision": raw["model"]["revision"],
                    "processor_revision": raw["model"]["processor_revision"],
                },
            },
        },
    }
    return Settings.model_validate(raw)


def _worker_settings(tmp_path: Path) -> Settings:
    settings = _base_settings(tmp_path)
    raw = settings.model_dump(mode="python")
    raw["model"]["revision"] = "model-commit-123"
    raw["model"]["processor_revision"] = "processor-commit-123"
    raw["deployment"] = {
        "mode": "split",
        "node": {"id": "training-01", "role": "training_worker"},
        "training": {
            "backend": "worker",
            "worker": {
                "inbox_directory": tmp_path / "inbox",
                "work_directory": tmp_path / "work",
                "result_directory": tmp_path / "results",
                "max_concurrent_jobs": 1,
                "retain_failed_jobs": True,
                "allowed_submitters": ["inference-01"],
            },
        },
    }
    return Settings.model_validate(raw)
