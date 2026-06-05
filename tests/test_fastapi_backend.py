from pathlib import Path
import os

from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from kagya.api.server import create_app
from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry
from kagya.memory import DualMemorySystem
from kagya.models import DummyProvider


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ADMIN_TOKEN = "test-admin-token"


class ThinkingProvider(DummyProvider):
    response_text = "<think>debug thought</think>Visible API answer."


def test_api_chat_works_with_dummy_provider_without_debug_leak(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/api/chat", json={"text": "hello", "attachments": [], "debug": False})

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"episode_id", "response", "emotion", "model"}
    assert data["response"] == "Visible API answer."
    assert "hidden_thought" not in data
    assert "prompt" not in data
    assert "<think>" not in str(data)


def test_api_chat_accepts_multiple_attachments(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/chat",
        json={
            "text": "describe these files",
            "attachments": [
                {"type": "image", "url": "file:///tmp/image.png", "name": "image.png"},
                {"type": "audio", "url": "file:///tmp/audio.wav", "duration_ms": 1200},
                {"type": "video", "url": "file:///tmp/video.mp4", "content_type": "video/mp4"},
            ],
            "debug": False,
        },
    )

    assert response.status_code == 200


def test_api_chat_accepts_legacy_message_key(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/api/chat", json={"message": "hello", "attachments": []})

    assert response.status_code == 200


def test_api_chat_debug_includes_hidden_thought_and_loss(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/chat/debug",
        headers=admin_headers(),
        json={"text": "hello", "attachments": [], "debug": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Visible API answer."
    assert data["hidden_thought"] == "debug thought"
    assert data["loss"] == DummyProvider.loss_value
    assert "prompt" in data
    assert "retrieved_memory" in data
    assert "generation_params" in data


def test_cors_middleware_uses_configured_origins(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)

    cors = next(middleware for middleware in app.user_middleware if middleware.cls is CORSMiddleware)
    assert cors.kwargs["allow_origins"] == settings.api.cors_origins


def test_adapter_endpoints_enforce_lifecycle_transitions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    registry.register_candidate(
        adapter_id="adapter-api",
        adapter_path=tmp_path / "adapter-api",
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash",
    )

    invalid = client.post("/api/adapters/adapter-api/activate", headers=admin_headers())
    assert invalid.status_code == 400

    evaluated = client.post(
        "/api/adapters/adapter-api/evaluate",
        headers=admin_headers(),
        json={"deterministic_score": 0.9},
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["status"] == "trial_active"
    approved = client.post("/api/adapters/adapter-api/approve", headers=admin_headers())
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    active = client.post("/api/adapters/adapter-api/activate", headers=admin_headers())
    assert active.status_code == 200
    assert active.json()["status"] == "active"
    listed = client.get("/api/adapters", headers=admin_headers())
    assert listed.status_code == 200
    assert listed.json()["adapters"][0]["status"] == "active"


def test_sleep_endpoint_returns_dry_run_result(tmp_path: Path) -> None:
    client = _client(tmp_path)
    memory = client.app.state.memory_system
    memory.save_episodic(
        "sleep input",
        "sleep output",
        hidden_thought="sleep thought",
        emotion_arousal=0.9,
    )

    response = client.post("/api/sleep/run", headers=admin_headers())

    assert response.status_code == 200
    data = response.json()
    assert data["selected_episode_ids"]
    assert data["semantic_memory_ids"]
    assert data["adapter_id"] is not None
    assert data["adapter_status"] == "candidate"
    assert data["dry_run"] is True


def test_memory_api_does_not_expose_hidden_thought(tmp_path: Path) -> None:
    client = _client(tmp_path)
    memory = client.app.state.memory_system
    episode_id = memory.save_episodic(
        "memory input",
        "memory output",
        hidden_thought="private memory thought",
    )

    search = client.get("/api/memory/search", headers=admin_headers(), params={"query": "memory"})
    detail = client.get(f"/api/memory/episodes/{episode_id}", headers=admin_headers())

    assert search.status_code == 200
    assert detail.status_code == 200
    assert "hidden_thought" not in str(search.json())
    assert "private memory thought" not in str(search.json())
    assert "hidden_thought" not in detail.json()
    assert "private memory thought" not in str(detail.json())


def test_sensitive_api_requires_admin_token(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.post("/api/chat", json={"text": "hello", "attachments": [], "debug": False}).status_code == 200
    assert client.post("/api/chat/debug", json={"text": "hello", "attachments": [], "debug": True}).status_code == 401
    assert client.get("/api/memory/search", params={"query": "hello"}).status_code == 401
    assert client.post("/api/sleep/run").status_code == 401
    assert client.get("/api/adapters").status_code == 401


def test_sensitive_api_reports_missing_admin_token_config(tmp_path: Path) -> None:
    client = _client(tmp_path, configure_admin_token=False)

    response = client.post("/api/chat/debug", headers=admin_headers(), json={"text": "hello", "attachments": []})

    assert response.status_code == 503
    assert "KAGYA_TEST_ADMIN_TOKEN" in response.json()["detail"]


def _client(tmp_path: Path, *, settings: Settings | None = None, configure_admin_token: bool = True) -> TestClient:
    if configure_admin_token:
        os.environ["KAGYA_TEST_ADMIN_TOKEN"] = ADMIN_TOKEN
    else:
        os.environ.pop("KAGYA_TEST_ADMIN_TOKEN", None)
    app_settings = settings or _settings(tmp_path)
    app = create_app(app_settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(app_settings)
    app.state.adapter_registry = AdapterRegistry(app_settings)
    return TestClient(app)


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_api_test",
                    "db2_collection": "cortex_api_test",
                }
            ),
            "sleep": settings.sleep.model_copy(
                update={"dream_dataset_path": tmp_path / "dreams" / "dream_dataset.jsonl"}
            ),
            "qlora": settings.qlora.model_copy(
                update={"output_dir": tmp_path / "adapters", "dry_run": True}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": [],
                }
            ),
            "api": settings.api.model_copy(update={"admin_token_env": "KAGYA_TEST_ADMIN_TOKEN"}),
        }
    )


def admin_headers() -> dict[str, str]:
    return {"X-KAGYA-Admin-Token": ADMIN_TOKEN}
