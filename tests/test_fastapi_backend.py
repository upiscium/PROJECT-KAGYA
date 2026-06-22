from pathlib import Path
import json
import os

from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from kagya.api.server import create_app
from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider
from kagya.tools import (
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutor,
    ToolRegistry,
    ToolStatus,
    ToolType,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ADMIN_TOKEN = "test-admin-token"


class ThinkingProvider(DummyProvider):
    response_text = "<think>debug thought</think>Visible API answer."


class EmptyFallbackProvider(DummyProvider):
    response_text = "<think>primary hidden only</think>"

    def __init__(self) -> None:
        self.last_model_id = "primary-model"
        self.last_fallback_used = False

    def generate_fallback(self, prompt: str) -> str:
        self.last_model_id = "fallback-model"
        self.last_fallback_used = True
        return "<think>fallback hidden only</think>"


class SuccessfulFallbackProvider(EmptyFallbackProvider):
    def generate_fallback(self, prompt: str) -> str:
        self.last_model_id = "fallback-model"
        self.last_fallback_used = True
        return "Fallback visible API answer."


def test_api_chat_works_with_dummy_provider_without_debug_leak(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/chat", json={"text": "hello", "attachments": [], "debug": False}
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"episode_id", "response", "emotion", "model"}
    assert data["response"] == "Visible API answer."
    assert data["model"]["fallback_used"] is False
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
                {
                    "type": "video",
                    "url": "file:///tmp/video.mp4",
                    "content_type": "video/mp4",
                },
            ],
            "debug": False,
        },
    )

    assert response.status_code == 200
    assert "prompt" not in response.json()


def test_api_chat_debug_includes_attachment_metadata_in_prompt(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/chat/debug",
        headers=admin_headers(),
        json={
            "text": "describe this file",
            "attachments": [
                {
                    "type": "image",
                    "url": "file:///tmp/image.png",
                    "name": "image.png",
                    "content_type": "image/png",
                    "duration_ms": 1200,
                }
            ],
            "debug": True,
        },
    )

    assert response.status_code == 200
    prompt = response.json()["prompt"]
    assert "Attachments:" in prompt
    assert "type=image" in prompt
    assert "name=image.png" in prompt
    assert "url=file:///tmp/image.png" in prompt
    assert "content_type=image/png" in prompt
    assert "duration_ms" not in prompt


def test_api_chat_accepts_legacy_message_key(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/api/chat", json={"message": "hello", "attachments": []}
    )

    assert response.status_code == 200


def test_api_chat_returns_500_when_fallback_has_no_visible_response(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    client.app.state.model_provider = EmptyFallbackProvider()

    response = client.post("/api/chat", json={"text": "hello", "attachments": []})

    assert response.status_code == 500
    assert "empty visible response" in response.json()["detail"]


def test_debug_chat_returns_500_when_fallback_has_no_visible_response(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    client.app.state.model_provider = EmptyFallbackProvider()

    response = client.post(
        "/api/chat/debug",
        headers=admin_headers(),
        json={"text": "hello", "attachments": [], "debug": True},
    )

    assert response.status_code == 500
    assert "empty visible response" in response.json()["detail"]


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


def test_system_info_exposes_safe_runtime_metadata(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/system/info")

    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "PROJECT-KAGYA"
    assert data["status"] == "ok"
    assert data["build"]["version"]
    assert data["runtime"] == {
        "environment": "development",
        "provider": "dummy",
        "primary_model_id": "google/gemma-4-E4B",
        "fallback_configured": True,
        "transformers_4bit": True,
        "qlora_dry_run": True,
        "admin_token_configured": True,
    }


def test_system_info_does_not_expose_secrets_or_private_paths(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/system/info")

    assert response.status_code == 200
    payload = response.text
    assert ADMIN_TOKEN not in payload
    assert "KAGYA_TEST_ADMIN_TOKEN" not in payload
    assert str(tmp_path) not in payload
    assert "hidden_thought" not in payload
    assert "prompt" not in payload


def test_system_events_require_admin_token(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/system/events")

    assert response.status_code == 401


def test_system_events_include_fallback_without_private_fields(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.app.state.model_provider = SuccessfulFallbackProvider()

    chat = client.post("/api/chat", json={"text": "hello", "attachments": []})
    events = client.get("/api/system/events", headers=admin_headers())

    assert chat.status_code == 200
    assert events.status_code == 200
    payload = events.json()
    assert payload["events"][-1]["category"] == "model"
    assert payload["events"][-1]["event_type"] == "fallback_used"
    assert payload["events"][-1]["metadata"]["model_id"] == "fallback-model"
    assert "hidden_thought" not in events.text
    assert "prompt" not in events.text
    assert ADMIN_TOKEN not in events.text


def test_system_events_include_sleep_and_adapter_lifecycle(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    memory = client.app.state.memory_system
    memory.save_episodic("sleep input", "sleep output", emotion_arousal=0.9)
    registry.register_candidate(
        adapter_id="adapter-observed",
        adapter_path=tmp_path / "adapter-observed",
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash",
    )

    sleep = client.post("/api/sleep/run", headers=admin_headers())
    evaluated = client.post(
        "/api/adapters/adapter-observed/evaluate",
        headers=admin_headers(),
        json={"deterministic_score": 0.9},
    )
    approved = client.post(
        "/api/adapters/adapter-observed/approve", headers=admin_headers()
    )
    events = client.get("/api/system/events", headers=admin_headers())

    assert sleep.status_code == 200
    assert evaluated.status_code == 200
    assert approved.status_code == 200
    event_pairs = {
        (event["category"], event["event_type"]) for event in events.json()["events"]
    }
    assert ("sleep", "run_completed") in event_pairs
    assert ("adapter", "evaluated") in event_pairs
    assert ("adapter", "approved") in event_pairs


def test_system_events_include_tool_audit_events(tmp_path: Path) -> None:
    client = _client(tmp_path)
    registry = ToolRegistry()
    registry.register_declared(
        ToolDefinition(
            name="safe_template",
            description="format text",
            tool_type=ToolType.TEXT_TEMPLATE,
            output_template="hello {name}",
            human_approved=True,
            status=ToolStatus.APPROVED,
        )
    )
    executor = ToolExecutor(registry)
    executor.execute(
        ToolExecutionRequest(tool_name="safe_template", arguments={"name": "operator"})
    )
    client.app.state.tool_executor = executor

    response = client.get("/api/system/events", headers=admin_headers())

    assert response.status_code == 200
    tool_events = [
        event for event in response.json()["events"] if event["category"] == "tool"
    ]
    assert tool_events[-1]["event_type"] == "executed"
    assert tool_events[-1]["metadata"] == {
        "tool_name": "safe_template",
        "status": "approved",
        "tool_type": "text_template",
    }


def test_cors_middleware_uses_configured_origins(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)

    cors = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls is CORSMiddleware
    )
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
    before_activation_chat = client.post(
        "/api/chat", json={"text": "before activation", "attachments": []}
    )
    assert before_activation_chat.status_code == 200
    assert before_activation_chat.json()["model"]["adapter_id"] is None

    active = client.post("/api/adapters/adapter-api/activate", headers=admin_headers())
    assert active.status_code == 200
    assert active.json()["status"] == "active"
    after_activation_chat = client.post(
        "/api/chat", json={"text": "after activation", "attachments": []}
    )
    assert after_activation_chat.status_code == 200
    assert after_activation_chat.json()["model"]["adapter_id"] == "adapter-api"
    listed = client.get("/api/adapters", headers=admin_headers())
    assert listed.status_code == 200
    assert listed.json()["adapters"][0]["status"] == "active"


def test_adapter_evaluation_reports_missing_eval_set_without_rejecting_candidate(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"eval_sets": [tmp_path / "missing_eval_set.json"]}
            )
        }
    )
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    registry.register_candidate(
        adapter_id="adapter-missing-eval",
        adapter_path=tmp_path / "adapter-missing-eval",
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash",
    )

    response = client.post(
        "/api/adapters/adapter-missing-eval/evaluate",
        headers=admin_headers(),
        json={},
    )

    assert response.status_code == 400
    assert "Configured eval set does not exist" in response.json()["detail"]
    assert registry.lookup("adapter-missing-eval").status.value == "candidate"


def test_evaluation_result_endpoints_list_and_return_json(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    result_dir = settings.adapter_registry.eval_result_dir
    result_dir.mkdir(parents=True)
    result_path = result_dir / "adapter-api.json"
    result_path.write_text(
        json.dumps(
            {
                "adapter_id": "adapter-api",
                "score": 0.9,
                "decision": "trial_active",
                "eval_sets": ["eval.json"],
                "case_count": 1,
                "prompt": "private prompt",
                "nested": {"hidden_thought": "private thought"},
            }
        ),
        encoding="utf-8",
    )

    listed = client.get("/api/evaluations", headers=admin_headers())
    detail = client.get("/api/evaluations/adapter-api.json", headers=admin_headers())

    assert listed.status_code == 200
    assert listed.json()["results"][0]["filename"] == "adapter-api.json"
    assert listed.json()["results"][0]["adapter_id"] == "adapter-api"
    assert listed.json()["results"][0]["score"] == 0.9
    assert detail.status_code == 200
    assert detail.json()["payload"]["decision"] == "trial_active"
    assert detail.json()["payload"]["prompt"] == "[redacted]"
    assert detail.json()["payload"]["nested"]["hidden_thought"] == "[redacted]"


def test_evaluation_result_endpoints_reject_unsafe_paths(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/evaluations/../config.yaml", headers=admin_headers())

    assert response.status_code == 404


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

    search = client.get(
        "/api/memory/search", headers=admin_headers(), params={"query": "memory"}
    )
    detail = client.get(f"/api/memory/episodes/{episode_id}", headers=admin_headers())

    assert search.status_code == 200
    assert detail.status_code == 200
    assert "hidden_thought" not in str(search.json())
    assert "private memory thought" not in str(search.json())
    assert "hidden_thought" not in detail.json()
    assert "private memory thought" not in str(detail.json())


def test_sensitive_api_requires_admin_token(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert (
        client.post(
            "/api/chat", json={"text": "hello", "attachments": [], "debug": False}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/chat/debug", json={"text": "hello", "attachments": [], "debug": True}
        ).status_code
        == 401
    )
    assert (
        client.get("/api/memory/search", params={"query": "hello"}).status_code == 401
    )
    assert client.post("/api/sleep/run").status_code == 401
    assert client.get("/api/adapters").status_code == 401


def test_sensitive_api_reports_missing_admin_token_config(tmp_path: Path) -> None:
    client = _client(tmp_path, configure_admin_token=False)

    response = client.post(
        "/api/chat/debug",
        headers=admin_headers(),
        json={"text": "hello", "attachments": []},
    )

    assert response.status_code == 503
    assert "KAGYA_TEST_ADMIN_TOKEN" in response.json()["detail"]


def _client(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    configure_admin_token: bool = True,
) -> TestClient:
    if configure_admin_token:
        os.environ["KAGYA_TEST_ADMIN_TOKEN"] = ADMIN_TOKEN
    else:
        os.environ.pop("KAGYA_TEST_ADMIN_TOKEN", None)
    app_settings = settings or _settings(tmp_path)
    app = create_app(app_settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(
        app_settings, embedding_function=DeterministicEmbeddingFunction()
    )
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
                update={
                    "dream_dataset_path": tmp_path / "dreams" / "dream_dataset.jsonl"
                }
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
            "api": settings.api.model_copy(
                update={"admin_token_env": "KAGYA_TEST_ADMIN_TOKEN"}
            ),
        }
    )


def admin_headers() -> dict[str, str]:
    return {"X-KAGYA-Admin-Token": ADMIN_TOKEN}
