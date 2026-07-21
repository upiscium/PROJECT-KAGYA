from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
import pytest

from kagya.api.server import create_app
from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import (
    AgentEventType,
    AgentStateStore,
    EventJournal,
    JournalLifecycle,
    StateWAL,
    hash_snapshot,
)
from kagya.tools import (
    ToolAuditEvent,
    ToolDefinition,
    ToolAuditLog,
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


class PreloadProvider(DummyProvider):
    def __init__(self) -> None:
        self.processor_loaded = False
        self.model_loaded = False

    def get_processor(self) -> object:
        self.processor_loaded = True
        return object()

    def get_model(self) -> object:
        self.model_loaded = True
        return object()


def test_api_chat_works_with_dummy_provider_without_debug_leak(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/chat", json={"text": "hello", "attachments": [], "debug": False}
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "context_id",
        "episode_id",
        "experience_id",
        "response",
        "emotion",
        "model",
    }
    assert data["response"] == "Visible API answer."
    assert data["experience_id"].startswith("experience-")
    assert data["model"]["fallback_used"] is False
    assert "hidden_thought" not in data
    assert "prompt" not in data
    assert "<think>" not in str(data)


def test_experience_api_exposes_structured_state_without_chat_content(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    chat = client.post(
        "/api/chat",
        json={
            "text": "private experience request",
            "attachments": [],
            "interlocutor_key": "person-one",
        },
    ).json()

    assert client.get("/api/experiences").status_code == 401
    listed = client.get("/api/experiences", headers=admin_headers())
    detail = client.get(
        f"/api/experiences/{chat['experience_id']}", headers=admin_headers()
    )

    assert listed.status_code == 200
    assert detail.status_code == 200
    assert listed.json()["experiences"][0] == detail.json()
    assert detail.json()["identity_origin"]["actor"] == "user"
    assert detail.json()["interlocutor_ids"] == ["person-one"]
    assert detail.json()["result_refs"]["memory"] == [f"episode:{chat['episode_id']}"]
    serialized = json.dumps(detail.json())
    assert "private experience request" not in serialized
    assert "Visible API answer" not in serialized
    assert "hidden_thought" not in serialized


def test_narrative_self_api_is_admin_only_and_returns_structured_state(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    assert client.get("/api/narrative-self").status_code == 401
    created = client.post(
        "/api/narrative-self/future-self",
        headers=admin_headers(),
        json={
            "projection_id": "planning-future",
            "description": "Become more capable at planning",
            "theme_codes": ["planning"],
            "desired_level": 0.9,
            "current_level": 0.3,
            "evidence_refs": ["identity-claim:planning"],
        },
    )
    inspected = client.get("/api/narrative-self", headers=admin_headers())

    assert created.status_code == 200
    assert created.json()["gap"] == pytest.approx(0.6)
    assert created.json()["related_motivation_ids"]
    assert inspected.status_code == 200
    assert inspected.json()["future_self"][0]["projection_id"] == "planning-future"
    assert "hidden_thought" not in json.dumps(inspected.json())
    assert "prompt" not in json.dumps(inspected.json())


def test_experience_state_survives_subject_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as first_client:
        created = first_client.post(
            "/api/chat", json={"text": "persist experience", "attachments": []}
        ).json()
        proposed_belief = first_client.post(
            "/api/beliefs",
            headers=admin_headers(),
            json={
                "belief_id": "persistent-belief",
                "experience_id": created["experience_id"],
                "proposition": "A reviewed persistent proposition",
            },
        )
        assert proposed_belief.status_code == 200
        assert (
            first_client.post(
                "/api/beliefs/persistent-belief/resolve",
                headers=admin_headers(),
                json={
                    "accept": True,
                    "confidence": 0.7,
                    "epistemic_status": "probable",
                    "reason_code": "reviewed",
                    "evidence_refs": [f"experience:{created['experience_id']}"],
                },
            ).status_code
            == 200
        )
    snapshot_text = settings.agent_state.path.read_text(encoding="utf-8")
    assert "persist experience" not in snapshot_text
    assert "hidden_thought" not in snapshot_text

    with _client(tmp_path, settings=settings) as restarted_client:
        restored = restarted_client.get(
            f"/api/experiences/{created['experience_id']}",
            headers=admin_headers(),
        )
        restored_beliefs = restarted_client.get(
            "/api/beliefs",
            params={"active_only": True},
            headers=admin_headers(),
        )

    assert restored.status_code == 200
    assert restored.json()["experience_id"] == created["experience_id"]
    assert restored_beliefs.json()["beliefs"][0]["belief_id"] == "persistent-belief"


def test_chat_creates_and_resumes_explicit_context(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first = client.post("/api/chat", json={"text": "first", "attachments": []})
    context_id = first.json()["context_id"]

    second = client.post(
        "/api/chat",
        json={"text": "continue", "attachments": [], "context_id": context_id},
    )

    assert second.status_code == 200
    assert second.json()["context_id"] == context_id
    episode = client.app.state.memory_system.get_episodic(second.json()["episode_id"])
    assert episode is not None
    assert episode.context_id == context_id
    assert episode.correlation_id == context_id
    assert episode.source_channel == "api.chat"
    completed = [
        event
        for event in client.app.state.runtime_event_log.recent()
        if event.category == "agent" and event.event_type == "completed"
    ]
    assert completed[-1].metadata["correlation_id"] == context_id


def test_chat_rejects_unknown_context(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/api/chat",
        json={"text": "continue", "context_id": "ctx-missing", "attachments": []},
    )

    assert response.status_code == 404


def test_chat_rejects_closed_context(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first = client.post("/api/chat", json={"text": "first", "attachments": []})
    context_id = first.json()["context_id"]
    suspended = client.post(
        f"/api/contexts/{context_id}/suspend", headers=admin_headers()
    )
    resumed = client.post(f"/api/contexts/{context_id}/resume", headers=admin_headers())
    ended = client.post(f"/api/contexts/{context_id}/end", headers=admin_headers())

    response = client.post(
        "/api/chat",
        json={"text": "continue", "context_id": context_id, "attachments": []},
    )

    assert suspended.json()["status"] == "suspended"
    assert resumed.json()["status"] == "active"
    assert ended.json()["status"] == "closed"
    assert response.status_code == 409


def test_concurrent_chat_requests_share_one_ordered_subject_state(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(
            executor.map(
                lambda index: client.post(
                    "/api/chat",
                    json={"text": f"message-{index}", "attachments": []},
                ),
                range(4),
            )
        )

    assert [response.status_code for response in responses] == [200] * 4
    assert client.app.state.main_loop.session_state.turns == []
    assert (
        len(client.app.state.main_loop.working_memory.items)
        <= client.app.state.settings.working_memory.item_capacity
    )
    assert (
        len(
            [
                item
                for item in client.app.state.main_loop.working_memory.items
                if item.reference and item.reference.startswith("episode:")
            ]
        )
        == 4
    )
    assert len(client.app.state.memory_system.db1.get()["ids"]) == 4
    completed = [
        event
        for event in client.app.state.runtime_event_log.recent()
        if event.category == "agent" and event.event_type == "completed"
    ]
    assert [event.metadata["processing_sequence"] for event in completed] == [
        1,
        2,
        3,
        4,
    ]
    assert all("text" not in event.metadata for event in completed)


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
    assert "source=file" in prompt
    assert "file:///tmp/image.png" not in prompt
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
    assert data["working_memory"]["item_capacity"] > 0
    assert data["working_memory"]["token_capacity"] > 0
    assert data["working_memory"]["items"]
    assert "rendered_content" not in str(data["working_memory"])
    assert data["loss_measurement"]["valid"] is True
    assert data["appraisal"]["novelty_valid"] is True
    assert data["emotion_update"]["valence_contributions"]


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
        "primary_model_id": load_settings(CONFIG_PATH).model.primary_id,
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


def test_system_events_include_safe_appraisal_components(tmp_path: Path) -> None:
    client = _client(tmp_path)
    chat = client.post("/api/chat", json={"text": "hello", "attachments": []})
    events = client.get("/api/system/events", headers=admin_headers())

    assert chat.status_code == 200
    appraisal_events = [
        event for event in events.json()["events"] if event["category"] == "appraisal"
    ]
    assert appraisal_events[-1]["metadata"]["measurement_valid"] is True
    assert "novelty" in appraisal_events[-1]["metadata"]
    assert "hello" not in str(appraisal_events[-1])
    assert "prompt" not in str(appraisal_events[-1])


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

    sleep = client.post(
        "/api/sleep/jobs", headers=admin_headers(), json={"idempotency_key": "events"}
    )
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
    assert ("sleep", "job_created") in event_pairs
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
        "reason": "executed text_template",
    }


def test_system_events_include_persisted_tool_audit_events(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    ToolAuditLog(settings.tools.audit_path).append(
        ToolAuditEvent(
            tool_name="missing",
            executed=False,
            status=None,
            tool_type=None,
            reason="Tool is not registered",
        )
    )

    response = client.get("/api/system/events", headers=admin_headers())

    assert response.status_code == 200
    tool_events = [
        event for event in response.json()["events"] if event["category"] == "tool"
    ]
    assert tool_events[-1]["event_type"] == "blocked"
    assert tool_events[-1]["metadata"] == {
        "tool_name": "missing",
        "status": None,
        "tool_type": None,
        "reason": "Tool is not registered",
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


def test_startup_preloads_transformers_provider(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"provider": "transformers"})}
    )
    provider = PreloadProvider()
    monkeypatch.setattr(
        "kagya.api.server.load_model_provider", lambda settings: provider
    )
    app = create_app(settings)
    app.state.memory_system = DualMemorySystem(
        settings, embedding_function=DeterministicEmbeddingFunction()
    )

    with TestClient(app):
        pass

    assert provider.processor_loaded is True
    assert provider.model_loaded is True
    assert app.state.model_provider is provider


def test_adapter_endpoints_enforce_lifecycle_transitions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    registry.register_candidate(
        adapter_id="adapter-api",
        adapter_path=tmp_path / "adapter-api",
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash",
        adapter_hash="hash-api",
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
    previous_loop = client.app.state.main_loop
    previous_turns = previous_loop.session_state.turns
    previous_emotion = previous_loop.emotion_engine

    active = client.post("/api/adapters/adapter-api/activate", headers=admin_headers())
    assert active.status_code == 200, active.text
    assert active.json()["status"] == "active"
    assert client.app.state.main_loop is not previous_loop
    assert client.app.state.main_loop.session_state.turns is previous_turns
    assert client.app.state.main_loop.emotion_engine is previous_emotion
    after_activation_chat = client.post(
        "/api/chat", json={"text": "after activation", "attachments": []}
    )
    assert after_activation_chat.status_code == 200
    assert after_activation_chat.json()["model"]["adapter_id"] == "adapter-api"
    assert after_activation_chat.json()["model"]["adapter_hash"] == "hash-api"
    assert after_activation_chat.json()["model"]["activation_sequence"] > 0
    runtime_state = client.get("/api/adapters/runtime", headers=admin_headers())
    assert runtime_state.json()["adapter_id"] == "adapter-api"
    assert runtime_state.json()["adapter_hash"] == "hash-api"
    provenance = client.get(
        "/api/adapters/adapter-api/provenance", headers=admin_headers()
    )
    assert provenance.status_code == 200
    assert provenance.json()["adapter"]["adapter_hash"] == "hash-api"
    assert provenance.json()["activation_history"][0]["action"] == "activate"
    listed = client.get("/api/adapters", headers=admin_headers())
    assert listed.status_code == 200
    assert listed.json()["adapters"][0]["status"] == "active"
    rolled_back = client.post("/api/adapters/rollback", headers=admin_headers())
    assert rolled_back.status_code == 200
    assert rolled_back.json()["action"] == "rollback"
    assert rolled_back.json()["adapter_id"] is None
    after_rollback_chat = client.post(
        "/api/chat", json={"text": "after rollback", "attachments": []}
    )
    assert after_rollback_chat.json()["model"]["adapter_id"] is None


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
                "previous_score": 0.7,
                "score_delta": 0.2,
                "regression": False,
                "decision": "trial_active",
                "status_before": "candidate",
                "status_after": "trial_active",
                "eval_sets": ["eval.json"],
                "case_count": 1,
                "prompt": "private prompt",
                "nested": {"hidden_thought": "private thought"},
            }
        ),
        encoding="utf-8",
    )

    listed = client.get("/api/evaluations", headers=admin_headers())
    history = client.get(
        "/api/evaluations/adapters/adapter-api/history", headers=admin_headers()
    )
    detail = client.get("/api/evaluations/adapter-api.json", headers=admin_headers())

    assert listed.status_code == 200
    assert listed.json()["results"][0]["filename"] == "adapter-api.json"
    assert listed.json()["results"][0]["adapter_id"] == "adapter-api"
    assert listed.json()["results"][0]["score"] == 0.9
    assert listed.json()["results"][0]["score_delta"] == 0.2
    assert listed.json()["results"][0]["status_after"] == "trial_active"
    assert history.status_code == 200
    assert history.json()["results"][0]["filename"] == "adapter-api.json"
    assert detail.status_code == 200
    assert detail.json()["payload"]["decision"] == "trial_active"
    assert detail.json()["payload"]["prompt"] == "[redacted]"
    assert detail.json()["payload"]["nested"]["hidden_thought"] == "[redacted]"


def test_evaluation_result_endpoints_reject_unsafe_paths(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/evaluations/../config.yaml", headers=admin_headers())

    assert response.status_code == 404


def test_sleep_endpoint_returns_persistent_async_job(tmp_path: Path) -> None:
    client = _client(tmp_path)
    memory = client.app.state.memory_system
    memory.save_episodic(
        "sleep input",
        "sleep output",
        hidden_thought="sleep thought",
        emotion_arousal=0.9,
    )

    started = time.monotonic()
    response = client.post(
        "/api/sleep/jobs",
        headers=admin_headers(),
        json={"idempotency_key": "sleep-api-test"},
    )

    assert response.status_code == 200
    assert time.monotonic() - started < 1.0
    job_id = response.json()["job_id"]
    data = _wait_for_sleep_job(client, job_id)
    assert data["status"] == "completed"
    assert data["selected_episode_ids"]
    assert data["semantic_memory_ids"]
    assert data["candidate_adapter_id"] is not None
    assert data["bundle_path"] is not None
    assert data["phase_started_at"]
    assert isinstance(data["phase_durations_seconds"], dict)
    assert "preparing" in data["phase_durations_seconds"]
    assert data["import_status"] == "completed"
    assert data["correlation_id"] == "sleep-api-test"
    nodes = client.get("/api/training/nodes", headers=admin_headers())
    assert nodes.status_code == 200
    assert nodes.json()["nodes"][0]["reachable"] is True
    duplicate = client.post(
        "/api/sleep/jobs",
        headers=admin_headers(),
        json={"idempotency_key": "sleep-api-test"},
    )
    assert duplicate.json()["job_id"] == job_id


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


def test_memory_api_archives_and_tags_records(tmp_path: Path) -> None:
    client = _client(tmp_path)
    memory = client.app.state.memory_system
    episode_id = memory.save_episodic("operator episode", "response")
    semantic_id = memory.save_semantic("operator semantic")

    tagged = client.post(
        f"/api/memory/episodes/{episode_id}/metadata",
        headers=admin_headers(),
        json={"tags": ["review", "keep"], "operator_metadata": {"owner": "ops"}},
    )
    archived = client.post(
        f"/api/memory/episodes/{episode_id}/archive", headers=admin_headers()
    )
    semantic_tagged = client.post(
        f"/api/memory/semantic/{semantic_id}/metadata",
        headers=admin_headers(),
        json={"tags": ["fact"]},
    )

    assert tagged.status_code == 200
    assert tagged.json()["tags"] == ["review", "keep"]
    assert tagged.json()["operator_metadata"] == {"owner": "ops"}
    assert archived.status_code == 200
    assert archived.json()["archived"] is True
    assert memory.db1.get(ids=[episode_id])["ids"] == [episode_id]
    assert semantic_tagged.status_code == 200
    assert semantic_tagged.json()["tags"] == ["fact"]


def test_agent_state_admin_snapshot_restore_and_reset(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/api/state/export").status_code == 401
    chat = client.post("/api/chat", json={"text": "stateful", "attachments": []})
    assert chat.status_code == 200

    exported = client.get("/api/state/export", headers=admin_headers())
    assert exported.status_code == 200
    snapshot = exported.json()
    assert snapshot["last_processed_event_sequence"] >= 2
    assert "turns" not in snapshot
    assert "stateful" not in str(snapshot)

    snapshot["emotion_state"] = {
        "valence": -0.25,
        "arousal": 0.4,
        "optimal_loss": 0.8,
    }
    restored = client.post("/api/state/restore", headers=admin_headers(), json=snapshot)
    assert restored.status_code == 200
    assert restored.json()["emotion_state"]["valence"] == -0.25

    reset = client.post("/api/state/reset", headers=admin_headers())
    assert reset.status_code == 200
    assert reset.json()["emotion_state"] == {
        "valence": 0.0,
        "arousal": 0.0,
        "optimal_loss": 1.0,
    }
    assert client.app.state.main_loop.session_state.turns == []


def test_point_in_time_restore_is_new_event_without_external_replay(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        baseline = client.app.state.state_wal.reconstruct(0)
        chat = client.post(
            "/api/chat", json={"text": "external side effect", "attachments": []}
        )
        assert chat.status_code == 200
        current = client.app.state.agent_state_store.last_snapshot
        assert current is not None
        current_hash = hash_snapshot(current)
        memory_ids = list(client.app.state.memory_system.db1.get()["ids"])

        reconstructed = client.get("/api/state/reconstruct/0", headers=admin_headers())
        dry_run = client.post("/api/state/restore/0/dry-run", headers=admin_headers())

        assert reconstructed.status_code == 200
        assert reconstructed.json()["snapshot_hash"] == baseline.snapshot_hash
        assert reconstructed.json()["external_side_effects_replayed"] is False
        assert dry_run.status_code == 200
        assert dry_run.json()["current_hash"] == current_hash
        assert (
            hash_snapshot(client.app.state.agent_state_store.last_snapshot)
            == current_hash
        )

        restored = client.post("/api/state/restore/0", headers=admin_headers())

        assert restored.status_code == 200
        assert (
            restored.json()["emotion_state"]
            == baseline.snapshot.model_dump(mode="json")["emotion_state"]
        )
        assert client.app.state.memory_system.db1.get()["ids"] == memory_ids

    records = StateWAL(settings.agent_state_wal.path).verify()
    assert records[-1].event_type == AgentEventType.STATE_POINT_IN_TIME_RESTORE.value
    assert records[-1].processing_sequence > baseline.sequence


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
    assert client.post("/api/sleep/jobs", json={}).status_code == 401
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


def test_value_admin_lifecycle_and_structured_evaluation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()

    assert client.get("/api/values").status_code == 401
    inspected = client.get("/api/values", headers=headers)
    assert inspected.status_code == 200
    assert {value["value_id"] for value in inspected.json()["values"]} == {
        "care",
        "honesty",
    }

    evaluated = client.post(
        "/api/values/evaluate",
        headers=headers,
        json={
            "options": {
                "gentle": {"care": 1.0, "honesty": 0.5},
                "blunt": {"care": -1.0, "honesty": 1.0},
            }
        },
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["options"][1]["conflicts"] == ["compassionate-honesty"]
    assert evaluated.json()["options"][0]["contributions"][0]["value_id"] == "care"

    update = {
        "proposal_id": "api-outcome-1",
        "kind": "outcome",
        "impacts": {"care": 1.0},
        "certainty": 1.0,
        "memory_ids": ["memory-api-1"],
        "source": "self",
    }
    updated = client.post("/api/values/updates", headers=headers, json=update)
    assert updated.status_code == 200
    assert updated.json()["updates"][0]["identity_origin"]["actor"] == "operator"
    assert updated.json()["updates"][0]["identity_origin"]["input_kind"] == "feedback"
    value_state = client.get("/api/values", headers=headers).json()["values"][0]
    assert value_state["origin_provenance"]["actor"] == "inherited"
    record = updated.json()["updates"][0]
    assert record["event_id"]
    assert record["event_sequence"] > 0
    assert record["memory_ids"] == ["memory-api-1"]
    assert record["evidence_ids"] == ["proposal:api-outcome-1:care"]
    revisions = client.get("/api/values/care/revisions", headers=headers)
    assert revisions.status_code == 200
    assert revisions.json()["revisions"][0]["revision_diff"]["changed_fields"][
        "weight"
    ][0] == pytest.approx(0.8)
    assert client.post("/api/values/updates", headers=headers, json=update).json() == {
        "updates": []
    }

    frozen = client.post(
        "/api/values/care/freeze", headers=headers, json={"frozen": True}
    )
    assert frozen.status_code == 200
    assert frozen.json()["frozen"] is True
    rejected = client.post(
        "/api/values/updates",
        headers=headers,
        json={**update, "proposal_id": "api-outcome-2"},
    )
    assert rejected.json()["updates"][0]["operation"] == "rejected"

    rolled_back = client.post(
        "/api/values/care/rollback",
        headers=headers,
        json={"target_revision": 1},
    )
    assert rolled_back.status_code == 200
    assert rolled_back.json()["frozen"] is False
    reset = client.post(
        "/api/values/reset", headers=headers, json={"value_ids": ["care"]}
    )
    assert reset.status_code == 200
    assert reset.json()["values"][0]["weight"] == 0.8


def test_experience_value_evidence_keeps_provenance_boundary(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()
    experience_id = client.post(
        "/api/chat", json={"text": "evidence", "attachments": []}
    ).json()["experience_id"]

    response = client.post(
        "/api/values/evidence/experience",
        headers=headers,
        json={
            "experience_id": experience_id,
            "impacts": {"care": 0.8},
            "proposal_id": "experience-evidence-api",
        },
    )

    assert response.status_code == 200
    update = response.json()["updates"][0]
    evidence_id = update["evidence_ids"][0]
    inspected = client.get("/api/values", headers=headers).json()
    care = next(item for item in inspected["values"] if item["value_id"] == "care")
    evidence = next(
        item for item in inspected["evidence"] if item["evidence_id"] == evidence_id
    )
    assert care["origin_provenance"]["actor"] == "inherited"
    assert care["origin_experience_ids"] == [experience_id]
    assert evidence["experience_ids"] == [experience_id]
    assert evidence["identity_origin"]["actor"] == "user"


def test_goal_and_commitment_admin_lifecycle(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()

    assert client.get("/api/goals").status_code == 401
    rejected_intrinsic = client.post(
        "/api/goals",
        headers=headers,
        json={
            "goal_type": "intrinsic",
            "description": "Operator cannot declare an intrinsic goal",
        },
    )
    assert rejected_intrinsic.status_code == 409
    intrinsic = client.post(
        "/api/goals",
        headers=headers,
        json={
            "goal_id": "intrinsic-goal",
            "goal_type": "external_request",
            "description": "Maintain internal coherence",
            "origin_value_id": "honesty",
            "priority": 0.3,
            "urgency": 0.3,
            "expected_utility": 0.5,
            "confidence": 0.8,
            "value_effects": {"honesty": 1.0},
        },
    )
    assert intrinsic.status_code == 200
    assert intrinsic.json()["goal_type"] == "external_request"
    assert intrinsic.json()["identity_origin"]["actor"] == "operator"
    assert intrinsic.json()["identity_origin"]["endorsement"] == "pending"
    adopted = client.post("/api/goals/intrinsic-goal/adopt", headers=headers)
    assert adopted.status_code == 200
    assert adopted.json()["action"] == "activate"
    adopted_goal = next(
        goal
        for goal in client.get("/api/goals", headers=headers).json()["goals"]
        if goal["goal_id"] == "intrinsic-goal"
    )
    assert adopted_goal["identity_origin"]["endorsement"] == "endorsed"

    external = client.post(
        "/api/goals",
        headers=headers,
        json={
            "goal_id": "external-goal",
            "goal_type": "external_request",
            "description": "Handle an urgent external request",
            "priority": 1.0,
            "urgency": 1.0,
            "expected_utility": 1.0,
            "confidence": 1.0,
            "conflict_ids": ["intrinsic-goal"],
        },
    )
    assert external.status_code == 200
    selected = client.post("/api/goals/external-goal/adopt", headers=headers)
    assert selected.status_code == 200
    assert selected.json()["conflicting_goal_ids"] == ["intrinsic-goal"]

    inspected = client.get("/api/goals", headers=headers).json()
    statuses = {goal["goal_id"]: goal["status"] for goal in inspected["goals"]}
    assert statuses == {
        "external-goal": "active",
        "intrinsic-goal": "suspended",
    }
    assert any(decision["action"] == "suspend" for decision in inspected["decisions"])
    decision_input = client.get("/api/goals/decision-input", headers=headers).json()
    assert decision_input["active_goals"][0]["goal_id"] == "external-goal"
    assert "no_action" in decision_input["allowed_actions"]

    valence_before = client.app.state.main_loop.emotion_engine.state.valence
    completed = client.post(
        "/api/goals/external-goal/transition",
        headers=headers,
        json={
            "status": "completed",
            "reason": "request_satisfied",
            "outcome": "response delivered",
        },
    )
    assert completed.status_code == 200
    assert completed.json()["transitions"][-1]["outcome"] == "response delivered"
    assert client.app.state.main_loop.emotion_engine.state.valence > valence_before

    reevaluated = client.post("/api/goals/reevaluate", headers=headers)
    assert reevaluated.status_code == 200
    assert any(
        decision["action"] == "resume" and decision["goal_id"] == "intrinsic-goal"
        for decision in reevaluated.json()["decisions"]
    )
    assert any(
        item.item_id == "goal:intrinsic-goal"
        for item in client.app.state.main_loop.working_memory.items
    )

    commitment = client.post(
        "/api/commitments",
        headers=headers,
        json={
            "commitment_id": "promise-1",
            "description": "Provide a follow-up",
            "value_effects": {"care": 0.5},
        },
    )
    assert commitment.status_code == 200
    assert commitment.json()["status"] == "active"
    assert commitment.json()["identity_origin"]["actor"] == "operator"
    assert commitment.json()["identity_origin"]["endorsement"] == "endorsed"
    assert "promise-1" in client.app.state.main_loop.self_model.state.commitment_refs
    fulfilled = client.post(
        "/api/commitments/promise-1/transition",
        headers=headers,
        json={
            "status": "fulfilled",
            "reason": "follow_up_provided",
            "outcome": "delivered",
        },
    )
    assert fulfilled.status_code == 200
    assert fulfilled.json()["status"] == "fulfilled"
    related_goal = client.app.state.main_loop.goal_manager.get("commitment:promise-1")
    assert related_goal.goal_type.value == "commitment"
    assert related_goal.status.value == "completed"

    expired_commitment = client.post(
        "/api/commitments",
        headers=headers,
        json={
            "commitment_id": "expired-promise",
            "description": "Already expired promise",
            "deadline": "2000-01-01T00:00:00Z",
        },
    )
    assert expired_commitment.status_code == 409
    assert (
        "expired-promise" not in client.app.state.main_loop.commitment_store.commitments
    )
    assert (
        "commitment:expired-promise"
        not in client.app.state.main_loop.goal_manager.goals
    )


def test_internal_motivation_forms_bounded_intrinsic_goal(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()
    first = client.post(
        "/api/chat", json={"text": "novel observation one", "attachments": []}
    )
    context_id = first.json()["context_id"]
    client.post(
        "/api/chat",
        json={
            "text": "novel observation two",
            "attachments": [],
            "context_id": context_id,
        },
    )
    client.post(
        "/api/chat",
        json={
            "text": "novel observation three",
            "attachments": [],
            "context_id": context_id,
        },
    )

    assert client.get("/api/motivation").status_code == 401
    state = client.get("/api/motivation", headers=headers)
    formed = client.post("/api/motivation/reevaluate", headers=headers)

    assert state.status_code == 200
    assert state.json()["records"]
    assert formed.status_code == 200
    assert 0 < len(formed.json()["goals"]) <= 2
    goal = formed.json()["goals"][0]
    assert goal["goal_type"] == "intrinsic"
    assert goal["identity_origin"]["actor"] == "self"
    assert goal["structured_target"]["motivation_id"]
    repeated = client.post("/api/motivation/reevaluate", headers=headers)
    assert repeated.json()["goals"] == []


def test_attention_admin_api_competes_and_controls_focus(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()
    chat = client.post(
        "/api/chat", json={"text": "private attention stimulus", "attachments": []}
    ).json()

    assert client.get("/api/attention").status_code == 401
    state = client.get("/api/attention", headers=headers)
    competed = client.post("/api/attention/compete", headers=headers)

    assert state.status_code == 200
    assert competed.status_code == 200
    candidate_id = f"experience:{chat['experience_id']}"
    assert candidate_id in {item["candidate_id"] for item in state.json()["candidates"]}
    refocused = client.post(
        "/api/attention/refocus",
        headers=headers,
        json={
            "candidate_ids": [candidate_id],
            "reason_code": "administrator_review",
            "provenance_refs": ["decision:attention-review"],
        },
    )
    deferred = client.post(
        f"/api/attention/{candidate_id}/defer",
        headers=headers,
        json={
            "reason_code": "defer_review",
            "provenance_refs": ["decision:attention-defer"],
        },
    )

    assert refocused.status_code == 200
    assert refocused.json()["candidate_ids"] == [candidate_id]
    assert deferred.status_code == 200
    resumed = client.post(f"/api/attention/{candidate_id}/resume", headers=headers)
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "available"
    final_state = client.get("/api/attention", headers=headers).json()
    candidate = next(
        item
        for item in final_state["candidates"]
        if item["candidate_id"] == candidate_id
    )
    assert candidate["status"] == "available"
    serialized = json.dumps(final_state)
    assert "private attention stimulus" not in serialized
    assert "hidden_thought" not in serialized
    assert "prompt" not in serialized


def test_decision_record_lifecycle_and_dataset_boundary(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()

    assert client.get("/api/decisions").status_code == 401
    context_response = client.post(
        "/api/chat", json={"text": "decision context", "attachments": []}
    ).json()
    context_id = context_response["context_id"]
    belief = client.post(
        "/api/beliefs",
        headers=headers,
        json={
            "belief_id": "decision-belief",
            "experience_id": context_response["experience_id"],
            "proposition": "The decision context is current",
            "subject": "decision-context",
            "predicate": "state",
            "object": "current",
            "context_scope": [context_id],
            "source_trust": 0.8,
            "confidence": 0.7,
        },
    )
    assert belief.status_code == 200
    assert belief.json()["lifecycle"] == "proposed"
    assert (
        client.get(
            "/api/beliefs", headers=headers, params={"active_only": True}
        ).json()["beliefs"]
        == []
    )
    resolved_belief = client.post(
        "/api/beliefs/decision-belief/resolve",
        headers=headers,
        json={
            "accept": True,
            "confidence": 0.9,
            "epistemic_status": "established",
            "reason_code": "reviewed_context_evidence",
            "evidence_refs": [f"experience:{context_response['experience_id']}"],
        },
    )
    assert resolved_belief.status_code == 200
    belief_context = client.post(
        "/api/chat/debug",
        headers=headers,
        json={
            "text": "use reviewed beliefs",
            "attachments": [],
            "context_id": context_id,
        },
    )
    assert belief_context.status_code == 200
    assert "Adopted belief (established" in belief_context.json()["prompt"]
    assert "The decision context is current" in belief_context.json()["prompt"]
    goal = client.post(
        "/api/goals",
        headers=headers,
        json={
            "goal_id": "decision-goal",
            "goal_type": "external_request",
            "description": "Make a traceable decision",
        },
    )
    assert goal.status_code == 200
    assert (
        client.post("/api/goals/decision-goal/adopt", headers=headers).status_code
        == 200
    )

    candidates = [
        {
            "candidate_id": "respond",
            "candidate_type": "respond",
            "proposed_action": "Provide an answer",
            "parameters": {"format": "text"},
            "prerequisites": [],
            "predicted_outcomes": [
                {
                    "outcome_id": "helpful",
                    "description": "The answer helps",
                    "probability": 1.0,
                    "utility": 0.8,
                }
            ],
            "uncertainty": 0.1,
            "estimated_cost": 0.1,
            "estimated_risk": 0.1,
            "value_effects": {"honesty": 0.5},
            "appraisal_contributions": {"goal_progress": 0.2},
        },
        {
            "candidate_id": "defer",
            "candidate_type": "defer",
            "proposed_action": "Wait for more evidence",
            "parameters": {},
            "prerequisites": [],
            "predicted_outcomes": [],
            "uncertainty": 0.2,
            "estimated_cost": 0.0,
            "estimated_risk": 0.0,
            "value_effects": {},
            "appraisal_contributions": {},
        },
    ]
    created = client.post(
        "/api/decisions",
        headers=headers,
        json={
            "decision_id": "decision-api-1",
            "context_id": context_id,
            "candidates": candidates,
        },
    )
    assert created.status_code == 200
    record = created.json()
    assert record["selected_candidate_id"] == "respond"
    assert record["status"] == "awaiting_outcome"
    assert record["actual_outcome"] is None
    assert record["triggering_event_id"]
    assert record["triggering_event_sequence"] > 0
    assert record["active_goal_ids"] == ["decision-goal"]
    assert set(record["value_revision_refs"]) == {"care", "honesty"}
    assert set(record["identity_origin_refs"]) == {
        "belief:decision-belief",
        "goal:decision-goal",
        "value:care",
        "value:honesty",
    }
    assert len(record["experience_refs"]) == 1
    assert record["belief_revision_refs"] == {"decision-belief": 1}
    assert "belief:decision-belief" in record["identity_origin_refs"]
    decision_experience = client.get(
        f"/api/experiences/{record['experience_refs'][0]}", headers=headers
    ).json()
    assert decision_experience["result_refs"]["decision"] == ["decision:decision-api-1"]
    assert set(record["emotion_snapshot"]) == {
        "valence",
        "arousal",
        "optimal_loss",
    }
    assert record["considered_candidates"][0]["value_contributions"]["honesty"] > 0

    awaiting = client.get(
        "/api/decisions", headers=headers, params={"status": "awaiting_outcome"}
    )
    assert len(awaiting.json()["decisions"]) == 1
    resolved = client.post(
        "/api/decisions/decision-api-1/outcome",
        headers=headers,
        json={
            "description": "The answer was partially useful",
            "utility": 0.2,
            "success": True,
        },
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "resolved"
    assert resolved.json()["prediction_error"] == pytest.approx(-0.6)
    assert resolved.json()["actual_outcome"]["observed_event_id"]
    value_snapshot = client.get("/api/values", headers=headers).json()
    assert value_snapshot["reassessments"][0]["decision_id"] == "decision-api-1"
    assert value_snapshot["reassessments"][0]["regret"] == pytest.approx(0.6)
    assert any(
        item["decision_id"] == "decision-api-1" for item in value_snapshot["evidence"]
    )

    dataset = client.get("/api/decisions/dataset", headers=headers)
    assert dataset.status_code == 200
    assert dataset.json()["records"][0]["source_id"] == "decision-api-1"
    assert "hidden_thought" not in json.dumps(dataset.json())

    client.app.state.model_provider.response_text = json.dumps(
        {"candidates": [candidates[1]]}
    )
    generated = client.post(
        "/api/decisions/generate",
        headers=headers,
        json={"situation": "Insufficient evidence"},
    )
    assert generated.status_code == 200
    assert generated.json()["candidates"][0]["candidate_type"] == "defer"
    assert "hidden_thought" not in json.dumps(generated.json())


def test_self_model_evidence_revision_and_decision_integration(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()

    assert client.get("/api/self-model").status_code == 401
    candidate = {
        "candidate_id": "no-op",
        "candidate_type": "no_op",
        "proposed_action": "Wait safely",
        "parameters": {"capability_ids": ["safe-waiting"]},
        "prerequisites": [],
        "predicted_outcomes": [
            {
                "outcome_id": "safe",
                "description": "No unsafe action",
                "probability": 1.0,
                "utility": 0.5,
            }
        ],
        "uncertainty": 0.1,
        "estimated_cost": 0.0,
        "estimated_risk": 0.0,
        "value_effects": {},
        "appraisal_contributions": {},
    }
    assert (
        client.post(
            "/api/decisions",
            headers=headers,
            json={"decision_id": "self-evidence", "candidates": [candidate]},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/decisions/self-evidence/outcome",
            headers=headers,
            json={"description": "Waited safely", "utility": 0.8, "success": True},
        ).status_code
        == 200
    )

    capability = client.post(
        "/api/self-model/capabilities/from-decision",
        headers=headers,
        json={
            "capability_id": "safe-waiting",
            "description": "Wait when action is unsafe",
            "decision_id": "self-evidence",
            "tags": ["safety"],
        },
    )
    assert capability.status_code == 200
    capability_state = capability.json()
    assert capability_state["capabilities"]["safe-waiting"]["confidence"] > 0.5
    assert (
        capability_state["capabilities"]["safe-waiting"]["evidence"][0]["source_id"]
        == "self-evidence"
    )

    limitation = client.post(
        "/api/self-model/limitations",
        headers=headers,
        json={
            "limitation_id": "no-remote",
            "description": "Cannot perform remote operations",
            "confidence": 1.0,
            "capability_ids": [],
            "tags": ["remote"],
            "evidence_refs": ["deployment:local"],
            "reason": "local deployment boundary",
        },
    )
    assert limitation.status_code == 200
    uncertainty = client.post(
        "/api/self-model/uncertainties",
        headers=headers,
        json={
            "uncertainty_id": "unknown-remote-state",
            "description": "Remote state is unknown",
            "confidence": 0.8,
            "tags": ["remote"],
            "evidence_refs": ["observation:missing"],
            "reason": "no observation",
        },
    )
    assert uncertainty.status_code == 200

    risky = {
        **candidate,
        "candidate_id": "remote-action",
        "candidate_type": "internal",
        "proposed_action": "Perform remote action",
        "parameters": {"topic_tags": ["remote"]},
        "predicted_outcomes": [
            {
                "outcome_id": "remote-success",
                "description": "Remote action succeeds",
                "probability": 1.0,
                "utility": 0.4,
            }
        ],
    }
    defer = {
        **candidate,
        "candidate_id": "defer",
        "candidate_type": "defer",
        "parameters": {},
    }
    evaluated = client.post(
        "/api/decisions",
        headers=headers,
        json={"decision_id": "self-evaluated", "candidates": [risky, defer]},
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["selected_candidate_id"] == "defer"
    risky_score = evaluated.json()["considered_candidates"][0]
    assert risky_score["self_model_contributions"] == {
        "limitation:no-remote": -0.5,
        "uncertainty:unknown-remote-state": pytest.approx(-0.32),
    }
    self_items = [
        item
        for item in client.app.state.main_loop.working_memory.items
        if item.kind.value == "self_model"
    ]
    assert len(self_items) == 2
    assert all("remote" in (item.content or "").lower() for item in self_items)

    original_summary = client.get("/api/self-model", headers=headers).json()["state"][
        "identity_summary"
    ]
    proposal = client.post(
        "/api/self-model/identity/proposals",
        headers=headers,
        json={
            "proposal_id": "self-claim",
            "proposed_summary": "An infallible remote operator",
            "proposed_traits": {"cautious": 1.0},
            "evidence_refs": [],
            "source": "self_report",
        },
    )
    assert proposal.status_code == 200
    assert proposal.json()["status"] == "pending"
    assert proposal.json()["source"] == "operator_proposal"
    assert proposal.json()["identity_origin"]["actor"] == "operator"
    assert proposal.json()["identity_origin"]["endorsement"] == "pending"
    assert "identity_summary_changed" in proposal.json()["contradictions"]
    assert (
        client.get("/api/self-model", headers=headers).json()["state"][
            "identity_summary"
        ]
        == original_summary
    )

    applied = client.post(
        "/api/self-model/identity/proposals/self-claim/resolve",
        headers=headers,
        json={"apply": True, "reason": "manual review"},
    )
    assert applied.status_code == 200
    assert applied.json()["identity_origin"]["endorsement"] == "endorsed"
    inspected = client.get("/api/self-model", headers=headers).json()
    assert inspected["state"]["traits"]["cautious"] == pytest.approx(0.1)
    assert inspected["history"][-1]["event_id"]
    assert "hidden_thought" not in json.dumps(inspected)


def test_agent_event_journal_commits_snapshot_without_private_chat_data(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        denied = client.get("/api/system/journal")
        assert denied.status_code == 401
        response = client.post(
            "/api/chat", json={"text": "private journal input", "attachments": []}
        )
        assert response.status_code == 200
        inspected = client.get("/api/system/journal", headers=admin_headers())
        assert inspected.status_code == 200
        assert "private journal input" not in inspected.text
        assert "debug thought" not in inspected.text

    records = EventJournal(settings.agent_journal.path).verify()
    snapshot = AgentStateStore(settings.agent_state.path).load(1.0)
    event_records = [
        record for record in records if record.event_type == AgentEventType.CHAT.value
    ]

    assert [record.lifecycle for record in event_records] == [
        JournalLifecycle.ACCEPTED,
        JournalLifecycle.STARTED,
        JournalLifecycle.PREPARED,
        JournalLifecycle.COMPLETED,
    ]
    assert event_records[-1].snapshot_hash == hash_snapshot(snapshot)
    serialized = settings.agent_journal.path.read_text(encoding="utf-8")
    assert "private journal input" not in serialized
    assert "debug thought" not in serialized


def test_agent_event_journal_continues_sequence_across_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        assert (
            client.post(
                "/api/chat", json={"text": "first", "attachments": []}
            ).status_code
            == 200
        )
    with _client(tmp_path, settings=settings) as restarted:
        assert (
            restarted.post(
                "/api/chat", json={"text": "second", "attachments": []}
            ).status_code
            == 200
        )

    records = EventJournal(settings.agent_journal.path).verify()
    assert [
        record.processing_sequence
        for record in records
        if record.lifecycle == JournalLifecycle.STARTED
    ] == [1, 2]


def test_operational_exports_are_private_persistent_and_traceable(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert client.get("/api/system/metrics").status_code == 401

        response = client.post(
            "/api/chat",
            json={"text": "do not export this private prompt", "attachments": []},
        )
        assert response.status_code == 200
        metrics = client.get("/api/system/metrics", headers=admin_headers())
        assert metrics.status_code == 200
        assert "kagya_agent_events_total" in metrics.text
        assert "kagya_storage_operation_seconds" in metrics.text
        assert "kagya_active_goals" in metrics.text
        assert "kagya_unresolved_decisions" in metrics.text
        assert "private prompt" not in metrics.text

        journal = client.get("/api/system/journal", headers=admin_headers()).json()
        event_id = next(
            record["event_id"]
            for record in journal["records"]
            if record["event_type"] == "chat"
        )
        traces = client.get(
            "/api/system/traces",
            params={"event_id": event_id},
            headers=admin_headers(),
        )
        assert traces.status_code == 200
        assert traces.json()["traces"][0]["event_id"] == event_id
        otlp = client.get("/api/system/telemetry", headers=admin_headers())
        assert otlp.status_code == 200
        assert otlp.json()["resourceMetrics"]

    persisted_before = settings.observability.metrics_path.read_text(encoding="utf-8")
    assert "do not export" not in persisted_before
    assert "debug thought" not in persisted_before
    assert "do not export" not in settings.observability.traces_path.read_text(
        encoding="utf-8"
    )
    with _client(tmp_path, settings=settings) as restarted:
        metrics_after = restarted.get(
            "/api/system/metrics", headers=admin_headers()
        ).text
        assert 'event_type="chat"' in metrics_after


def test_readiness_fails_when_subject_runtime_stops(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        client.app.state.agent_runtime.shutdown()
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert response.json()["checks"]["agent_runtime"] is False


def test_agent_event_journal_rejects_snapshot_hash_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        assert (
            client.post(
                "/api/chat", json={"text": "commit", "attachments": []}
            ).status_code
            == 200
        )
    store = AgentStateStore(settings.agent_state.path)
    snapshot = store.load(1.0)
    store.save(
        snapshot.model_copy(
            update={
                "emotion_state": snapshot.emotion_state.model_copy(
                    update={"valence": 0.75}
                )
            }
        )
    )

    with pytest.raises(RuntimeError, match="hashes disagree"):
        with _client(tmp_path, settings=settings):
            pass


def test_optional_admin_auth_preserves_disabled_token_only_behavior(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/state/snapshot",
        headers={**admin_headers(), "Origin": "https://untrusted.example"},
    )

    assert response.status_code == 200


def test_admin_auth_rejects_cross_site_and_enforces_roles(tmp_path: Path) -> None:
    settings = _auth_settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        client.cookies.update(
            {"kagya_admin_session": "session", "kagya_admin_csrf": "csrf"}
        )
        cross_site = client.post(
            "/api/state/snapshot",
            headers=_actor_headers("full_admin", origin="https://evil.example"),
        )
        read_only = client.post(
            "/api/state/snapshot",
            headers=_actor_headers("read_only"),
        )
        approval_training = client.post(
            "/api/sleep/jobs",
            headers=_actor_headers("approval_only"),
            json={},
        )
        approval_action = client.post(
            "/api/adapters/missing-adapter/reject",
            headers=_actor_headers("approval_only"),
        )
        readable = client.get(
            "/api/state/export", headers=_actor_headers("read_only", mutation=False)
        )

    assert cross_site.status_code == 403
    assert read_only.status_code == 403
    assert approval_training.status_code == 403
    assert approval_action.status_code != 403
    assert readable.status_code == 200


def test_admin_auth_requires_csrf_and_recent_reauthentication_for_reset(
    tmp_path: Path,
) -> None:
    settings = _auth_settings(tmp_path)
    cookies = {"kagya_admin_session": "session", "kagya_admin_csrf": "csrf"}
    with _client(tmp_path, settings=settings) as client:
        client.cookies.update(cookies)
        missing_csrf = client.post(
            "/api/state/snapshot",
            headers={
                **admin_headers(),
                "X-KAGYA-Actor": "alice",
                "X-KAGYA-Role": "full_admin",
                "Origin": "http://localhost:3000",
                "Sec-Fetch-Site": "same-origin",
            },
        )
        stale_auth = client.post(
            "/api/state/reset",
            headers=_actor_headers("full_admin"),
        )
        reset = client.post(
            "/api/state/reset",
            headers={
                **_actor_headers("full_admin"),
                "X-KAGYA-Reauthenticated-At": str(time.time()),
            },
        )

    assert missing_csrf.status_code == 403
    assert stale_auth.status_code == 403
    assert reset.status_code == 200
    records = EventJournal(settings.agent_journal.path).verify()
    audit = [record for record in records if record.lifecycle == JournalLifecycle.AUDIT]
    assert audit[-1].actor_id == "alice"
    assert audit[-1].actor_role == "full_admin"
    assert audit[-1].target == "POST /api/state/reset"
    journal_text = settings.agent_journal.path.read_text(encoding="utf-8")
    assert ADMIN_TOKEN not in journal_text
    assert "csrf" not in journal_text.lower()


def test_admin_auth_allows_token_only_loopback_recovery(tmp_path: Path) -> None:
    with _client(tmp_path, settings=_auth_settings(tmp_path)) as client:
        response = client.post("/api/state/snapshot", headers=admin_headers())

    assert response.status_code == 200


def test_structured_feedback_propagates_and_is_idempotent(tmp_path: Path) -> None:
    client = _client(tmp_path)
    chat = client.post(
        "/api/chat", json={"text": "Paris is in Germany", "attachments": []}
    ).json()
    body = {
        "idempotency_key": "feedback-operation-1",
        "feedback_id": "feedback-api-1",
        "target": {
            "target_type": "response",
            "target_id": chat["episode_id"],
            "episode_id": chat["episode_id"],
            "experience_id": chat["experience_id"],
            "context_id": chat["context_id"],
        },
        "signals": ["factual_error", "correction", "do_not_remember"],
        "correction": "Paris is in France.",
    }

    created = client.post("/api/feedback", json=body)
    duplicate = client.post("/api/feedback", json=body)

    assert created.status_code == 200
    assert duplicate.status_code == 200
    assert duplicate.json() == created.json()
    record = created.json()
    assert record["current_revision"] == 1
    current = record["revisions"][0]
    assert current["target"]["target_id"] == chat["episode_id"]
    assert current["target"]["experience_id"] == chat["experience_id"]
    assert current["provenance"]["actor_type"] == "user"
    assert current["provenance"]["event_id"]
    assert current["propagation"]["training_disposition"] == "exclude"
    assert current["propagation"]["value_evidence"]["status"] == "proposed"
    assert "reward" not in json.dumps(record).lower()
    assert "Paris is in France" not in json.dumps(record)

    original = client.app.state.memory_system.get_episodic(chat["episode_id"])
    correction_id = current["correction_memory_id"]
    correction = client.app.state.memory_system.get_episodic(correction_id)
    assert original is not None
    assert original.lifecycle_status.value == "rejected"
    assert original.training_included is False
    assert original.corrected_by_id == correction_id
    assert correction is not None
    assert correction.response == "Paris is in France."
    assert correction.supersedes_id == original.id
    retrieved = client.app.state.memory_system.retrieve_context("Paris Germany")
    assert original.id not in {item.id for item in retrieved.db1_results}

    assert client.get("/api/feedback").status_code == 401
    audited = client.get("/api/feedback", headers=admin_headers())
    assert audited.status_code == 200
    assert [item["feedback_id"] for item in audited.json()["feedback"]] == [
        "feedback-api-1"
    ]


def test_feedback_revision_and_withdrawal_restore_owned_effects(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    chat = client.post(
        "/api/chat", json={"text": "revise feedback", "attachments": []}
    ).json()
    created = client.post(
        "/api/feedback",
        json={
            "idempotency_key": "feedback-create",
            "feedback_id": "feedback-revise",
            "target": {
                "target_type": "response",
                "target_id": chat["episode_id"],
                "episode_id": chat["episode_id"],
                "experience_id": chat["experience_id"],
                "context_id": chat["context_id"],
            },
            "signals": ["bad", "exclude_from_training"],
        },
    )
    assert created.status_code == 200

    revised = client.post(
        "/api/feedback/feedback-revise/revisions",
        headers=admin_headers(),
        json={
            "idempotency_key": "feedback-revision",
            "expected_revision": 1,
            "signals": ["good", "remember"],
        },
    )
    assert revised.status_code == 200
    assert revised.json()["current_revision"] == 2
    memory = client.app.state.memory_system.get_episodic(chat["episode_id"])
    assert memory is not None
    assert memory.lifecycle_status.value == "active"
    assert memory.training_included is True

    withdrawn = client.post(
        "/api/feedback/feedback-revise/withdraw",
        headers=admin_headers(),
        json={
            "idempotency_key": "feedback-withdrawal",
            "expected_revision": 2,
        },
    )
    duplicate = client.post(
        "/api/feedback/feedback-revise/withdraw",
        headers=admin_headers(),
        json={
            "idempotency_key": "feedback-withdrawal",
            "expected_revision": 2,
        },
    )
    assert withdrawn.status_code == 200
    assert duplicate.json() == withdrawn.json()
    assert withdrawn.json()["current_revision"] == 3
    assert withdrawn.json()["revisions"][-1]["status"] == "withdrawn"
    assert len(withdrawn.json()["revisions"]) == 3


def test_public_feedback_enforces_response_provenance(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first = client.post("/api/chat", json={"text": "first", "attachments": []}).json()
    second = client.post("/api/chat", json={"text": "second", "attachments": []}).json()

    response = client.post(
        "/api/feedback",
        json={
            "idempotency_key": "cross-context",
            "target": {
                "target_type": "response",
                "target_id": first["episode_id"],
                "episode_id": first["episode_id"],
                "experience_id": first["experience_id"],
                "context_id": second["context_id"],
            },
            "signals": ["good"],
        },
    )

    assert response.status_code == 409
    assert "does not own" in response.json()["detail"]


def test_admin_feedback_updates_decision_outcome_value_proposal_and_dataset(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    decision = client.post(
        "/api/decisions",
        headers=admin_headers(),
        json={
            "decision_id": "feedback-decision",
            "candidates": [
                {
                    "candidate_id": "defer",
                    "candidate_type": "defer",
                    "proposed_action": "Wait",
                    "parameters": {},
                    "prerequisites": [],
                    "predicted_outcomes": [],
                    "uncertainty": 0.0,
                    "estimated_cost": 0.0,
                    "estimated_risk": 0.0,
                    "value_effects": {"honesty": 0.5},
                    "appraisal_contributions": {},
                }
            ],
        },
    )
    assert decision.status_code == 200

    feedback = client.post(
        "/api/feedback/admin",
        headers=admin_headers(),
        json={
            "idempotency_key": "decision-feedback",
            "feedback_id": "feedback-decision-outcome",
            "target": {
                "target_type": "decision",
                "target_id": "feedback-decision",
            },
            "signals": ["bad", "exclude_from_training"],
        },
    )

    assert feedback.status_code == 200
    propagation = feedback.json()["revisions"][0]["propagation"]
    assert propagation["decision_outcome_applied"] is True
    assert propagation["prediction_error"] == pytest.approx(-0.75)
    assert propagation["value_evidence"]["value_impacts"] == {
        "honesty": pytest.approx(-0.375)
    }
    stored = client.get("/api/decisions", headers=admin_headers()).json()["decisions"][
        0
    ]
    assert stored["actual_outcome"]["feedback_id"] == "feedback-decision-outcome"
    assert stored["training_included"] is False
    assert client.get("/api/decisions/dataset", headers=admin_headers()).json() == {
        "records": []
    }


def test_feedback_ledger_survives_restart_without_correction_text_in_snapshot(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as first_client:
        chat = first_client.post(
            "/api/chat", json={"text": "persistent feedback", "attachments": []}
        ).json()
        created = first_client.post(
            "/api/feedback",
            json={
                "idempotency_key": "persistent-feedback-operation",
                "feedback_id": "persistent-feedback",
                "target": {
                    "target_type": "response",
                    "target_id": chat["episode_id"],
                    "episode_id": chat["episode_id"],
                    "experience_id": chat["experience_id"],
                    "context_id": chat["context_id"],
                },
                "signals": ["correction"],
                "correction": "Private corrected answer",
            },
        )
        assert created.status_code == 200

    snapshot = settings.agent_state.path.read_text(encoding="utf-8")
    assert "persistent-feedback" in snapshot
    assert "Private corrected answer" not in snapshot
    assert "hidden_thought" not in snapshot
    with _client(tmp_path, settings=settings) as restarted_client:
        restored = restarted_client.get(
            "/api/feedback/persistent-feedback", headers=admin_headers()
        )

    assert restored.status_code == 200
    assert restored.json()["current_revision"] == 1


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
                    "dream_dataset_path": tmp_path / "dreams" / "dream_dataset.jsonl",
                    "job_registry_path": tmp_path / "training_jobs.json",
                    "training_artifact_directory": tmp_path / "training_artifacts",
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
            "tools": settings.tools.model_copy(
                update={
                    "path": tmp_path / "tool_registry.json",
                    "audit_path": tmp_path / "tool_audit.jsonl",
                }
            ),
            "agent_state": settings.agent_state.model_copy(
                update={"path": tmp_path / "agent_state.json"}
            ),
            "agent_journal": settings.agent_journal.model_copy(
                update={"path": tmp_path / "agent_journal.jsonl"}
            ),
            "agent_state_wal": settings.agent_state_wal.model_copy(
                update={"path": tmp_path / "private" / "agent_state_wal.jsonl"}
            ),
            "observability": settings.observability.model_copy(
                update={
                    "metrics_path": tmp_path / "operational_metrics.json",
                    "traces_path": tmp_path / "operational_traces.json",
                }
            ),
            "api": settings.api.model_copy(
                update={"admin_token_env": "KAGYA_TEST_ADMIN_TOKEN"}
            ),
        }
    )


def _auth_settings(tmp_path: Path) -> Settings:
    settings = _settings(tmp_path)
    return settings.model_copy(
        update={
            "api": settings.api.model_copy(
                update={
                    "admin_auth": settings.api.admin_auth.model_copy(
                        update={"enabled": True}
                    )
                }
            )
        }
    )


def _actor_headers(
    role: str,
    *,
    origin: str = "http://localhost:3000",
    mutation: bool = True,
) -> dict[str, str]:
    headers = {
        **admin_headers(),
        "X-KAGYA-Actor": "alice",
        "X-KAGYA-Role": role,
    }
    if mutation:
        headers.update(
            {
                "Origin": origin,
                "Sec-Fetch-Site": "same-origin",
                "X-KAGYA-CSRF-Token": "csrf",
            }
        )
    return headers


def admin_headers() -> dict[str, str]:
    return {"X-KAGYA-Admin-Token": ADMIN_TOKEN}


def test_autonomy_api_persists_and_processes_operator_wakeup(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "autonomy": settings.autonomy.model_copy(update={"enabled": False})
        }
    )
    with _client(tmp_path, settings=settings) as client:
        assert client.get("/api/autonomy/status").status_code == 401
        created = client.post(
            "/api/autonomy/wake-ups",
            headers=admin_headers(),
            json={
                "schedule_id": "api-operator-wake",
                "kind": "operator",
                "wake_at": "2020-01-01T00:00:00+00:00",
            },
        )
        assert created.status_code == 200
        scheduler = client.app.state.subject_scheduler
        cycle = scheduler.run_cycle()
        status_response = client.get(
            "/api/autonomy/status", headers=admin_headers()
        )

    assert cycle.result.value == "processed"
    assert status_response.status_code == 200
    assert status_response.json()["pending_count"] == 0
    snapshot = json.loads(settings.agent_state.path.read_text(encoding="utf-8"))
    schedules = snapshot["extensions"]["subject_scheduler"]["schedules"]
    assert schedules[0]["status"] == "completed"
    records = EventJournal(settings.agent_journal.path).verify()
    assert {record.event_type for record in records} >= {
        "autonomy_schedule",
        "autonomy_wake",
    }


def _wait_for_sleep_job(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/api/sleep/jobs/{job_id}", headers=admin_headers())
        data = response.json()
        if data["status"] in {"completed", "failed", "cancelled"}:
            return data
        time.sleep(0.01)
    raise AssertionError("sleep job did not finish")
