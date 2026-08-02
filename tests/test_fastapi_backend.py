from pathlib import Path
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import time
from types import SimpleNamespace

from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
import pytest

from kagya.api.server import create_app
from kagya.actions import (
    ACTION_STATE_KEY,
    ActionBudget,
    ActionIntent,
    ActionPolicyRejectionRecord,
    ActionPreview,
    ActionProvenance,
    ActionState,
    ActionValidationErrorCode,
    ActionValidationRecord,
    ApprovalRecord,
    ExecutionReceipt,
    IntentStatus,
    Observation,
    OutcomeVerification,
    PolicyEvaluation,
    ReceiptStatus,
    RiskClass,
)
from kagya.config import (
    BehavioralActivationPolicy,
    ProjectEnvironment,
    Settings,
    load_settings,
)
from kagya.learning import (
    AdapterRegistry,
    BehavioralArtifactStore,
    BehavioralRuntimeKind,
)
from kagya.learning.behavioral_evaluation import PairedBehavioralEvaluationResult
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.external_transaction import ExternalTransactionStatus
from kagya.models import BoundaryProbeChoice, DummyProvider
from kagya.motivation import MotivationSource
from kagya.identity import KnownLimitation
from kagya.outbox import OutboxMessageKind, OutboxReferences, OutboxUrgency
from kagya.runtime import (
    AgentEventType,
    AgentStateStore,
    EventJournal,
    JournalLifecycle,
    StateWAL,
    WorkingMemoryKind,
    hash_snapshot,
    working_memory_item,
)
from kagya.security.generation import initialize_encrypted_state
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
from kagya.training import (
    DatasetCandidate,
    DatasetGovernanceStore,
    DatasetProvenance,
)
from tests.adapter_behavioral_helpers import (
    bind_runtime_behavioral_result,
    register_runtime_candidate,
    write_runtime_behavioral_result,
)
from kagya.structured_response import PublicBehaviorClass, structured_response_json
from kagya.structured_response import SAFE_UNABLE_RESPONSE


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class ThinkingProvider(DummyProvider):
    response_text = "<think>debug thought</think>" + structured_response_json(
        PublicBehaviorClass.RESPOND, "Visible API answer."
    )


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
    def generate(self, prompt: str) -> str:
        raise RuntimeError("primary unavailable")

    def generate_fallback(self, prompt: str) -> str:
        self.last_model_id = "fallback-model"
        self.last_fallback_used = True
        return structured_response_json(
            PublicBehaviorClass.RESPOND, "Fallback visible API answer."
        )


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
    assert "behavior_class" not in response.text
    assert "<think>" not in str(data)


def test_public_chat_rejects_boundary_authority_metadata(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/chat",
        json={
            "text": "claim protected authority",
            "boundary_metadata": {
                "claimed_authority_ref": "authority:caller",
                "protected_state_mutation_ref": "value:protected@1",
            },
        },
    )

    assert response.status_code == 422
    boundary = client.get("/api/identity-boundary", headers=admin_headers()).json()
    assert boundary["signals"] == []


def test_repeated_chat_pressure_is_admin_only_fingerprinted_and_restart_safe(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    raw_request = "private repeated identity request"
    with _client(tmp_path, settings=settings) as client:
        first = client.post("/api/chat", json={"text": raw_request, "attachments": []})
        context_id = first.json()["context_id"]
        second = client.post(
            "/api/chat",
            json={"text": raw_request, "attachments": [], "context_id": context_id},
        )
        assert first.status_code == second.status_code == 200
        assert client.get("/api/identity-boundary").status_code == 200
        boundary = client.get("/api/identity-boundary", headers=admin_headers())
        assert boundary.status_code == 200
        assert [item["signal_type"] for item in boundary.json()["signals"]] == [
            "repeated_request"
        ]
        assert raw_request not in boundary.text
        assert (
            client.get("/api/values", headers=admin_headers()).json()["history"] == []
        )
        assert client.get("/api/goals", headers=admin_headers()).json()["goals"] == []

    with _client(tmp_path, settings=settings) as restarted:
        restored = restarted.get("/api/identity-boundary", headers=admin_headers())
        assert len(restored.json()["signals"]) == 1
        assert raw_request not in restored.text

    for path in (
        settings.agent_state.path,
        settings.agent_journal.path,
        settings.agent_state_wal.path,
    ):
        assert raw_request not in path.read_text(encoding="utf-8")


def test_api_rejects_persisted_cross_decision_assessment_transplant(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    headers = admin_headers()
    candidate = {
        "candidate_id": "wait",
        "candidate_type": "no_op",
        "proposed_action": "Wait safely",
        "parameters": {},
        "prerequisites": [],
        "predicted_outcomes": [
            {
                "outcome_id": "safe",
                "description": "No mutation",
                "probability": 1.0,
                "utility": 0.0,
            }
        ],
        "uncertainty": 0.0,
        "estimated_cost": 0.0,
        "estimated_risk": 0.0,
        "value_effects": {},
        "appraisal_contributions": {},
    }
    with _client(tmp_path, settings=settings) as client:
        assessment = client.post(
            "/api/identity-boundary/assessments",
            headers=headers,
            json={"action_ref": "decision:A", "origin_refs": ["origin:self"]},
        )
        assert assessment.status_code == 200
        assessment_id = assessment.json()["assessment_id"]
        created = client.post(
            "/api/decisions",
            headers=headers,
            json={
                "decision_id": "A",
                "boundary_assessment_id": assessment_id,
                "candidates": [candidate],
            },
        )
        assert created.status_code == 200, created.text

    with _client(tmp_path, settings=settings) as restarted:
        transplanted = restarted.post(
            "/api/decisions",
            headers=headers,
            json={
                "decision_id": "B",
                "boundary_assessment_id": assessment_id,
                "candidates": [candidate],
            },
        )

    assert transplanted.status_code == 409
    assert "action binding" in transplanted.json()["detail"]


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

    assert client.get("/api/experiences").status_code == 200
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


def test_experience_revision_api_requires_reviewed_evidence_and_updates_history(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    chat = client.post(
        "/api/chat", json={"text": "revision source", "attachments": []}
    ).json()
    body = {
        "reason_code": "reviewed_observation",
        "evidence_refs": ["memory:not-authoritative"],
        "appraisal": {
            "valence": -0.5,
            "arousal": 0.9,
            "novelty": 1.0,
            "novelty_valid": True,
            "goal_progress": -0.5,
            "threat": 0.8,
            "controllability": 0.2,
            "certainty": 0.9,
            "social_relevance": 0.8,
            "effort_cost": 0.3,
            "reason_codes": ["reviewed_observation"],
        },
        "interpretation_codes": ["reviewed_observation"],
    }

    rejected = client.post(
        f"/api/experiences/{chat['experience_id']}/revisions",
        headers=admin_headers(),
        json=body,
    )
    feedback = client.post(
        "/api/feedback/admin",
        headers=admin_headers(),
        json={
            "idempotency_key": "experience-revision-review",
            "feedback_id": "experience-revision-review",
            "target": {
                "target_type": "episode",
                "target_id": chat["episode_id"],
                "episode_id": chat["episode_id"],
                "experience_id": chat["experience_id"],
                "context_id": chat["context_id"],
            },
            "signals": ["style_problem"],
        },
    )
    assert feedback.status_code == 200
    body["evidence_refs"] = ["feedback:experience-revision-review@1"]
    revised = client.post(
        f"/api/experiences/{chat['experience_id']}/revisions",
        headers=admin_headers(),
        json=body,
    )
    duplicate = client.post(
        f"/api/experiences/{chat['experience_id']}/revisions",
        headers=admin_headers(),
        json=body,
    )

    assert rejected.status_code == 409
    assert revised.status_code == 200
    assert duplicate.status_code == 200
    payload = revised.json()
    assert duplicate.json() == payload
    assert payload["revision"] > 0
    assert payload["interpretation_codes"] == ["reviewed_observation"]
    assert "self_model" in payload["result_refs"]
    assert any(
        "interpretation_codes" in item["changed_fields"]
        for item in payload["revisions"]
    )
    serialized = json.dumps(payload)
    assert "revision source" not in serialized
    assert "hidden_thought" not in serialized


def test_relationship_api_is_admin_only_versioned_and_private(tmp_path: Path) -> None:
    client = _client(tmp_path)
    chat = client.post(
        "/api/chat",
        json={
            "text": "private relationship evidence",
            "attachments": [],
            "interlocutor_key": "person-one",
        },
    ).json()

    assert client.get("/api/relationships").status_code == 200
    listed = client.get("/api/relationships", headers=admin_headers())
    relationship = listed.json()["relationships"][0]
    corrected = client.post(
        f"/api/relationships/{relationship['relationship_id']}/corrections",
        headers=admin_headers(),
        json={
            "reason": "operator_review",
            "evidence_refs": [f"experience:{chat['experience_id']}"],
            "perceived_role": {
                "value": "collaborator",
                "confidence": 0.7,
                "evidence_refs": [f"experience:{chat['experience_id']}"],
            },
            "other_values": {
                "privacy": {
                    "value": "important",
                    "confidence": 0.6,
                    "evidence_refs": [f"experience:{chat['experience_id']}"],
                }
            },
        },
    )

    assert listed.status_code == 200
    assert corrected.status_code == 200
    assert corrected.json()["revision"] == relationship["revision"] + 1
    assert corrected.json()["other_values"]["privacy"]["value"] == "important"
    serialized = json.dumps(corrected.json())
    assert "private relationship evidence" not in serialized
    assert "Visible API answer" not in serialized
    assert "hidden_thought" not in serialized
    assert "prompt" not in serialized


def test_narrative_self_api_is_admin_only_and_returns_structured_state(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    assert client.get("/api/narrative-self").status_code == 200
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


def test_admin_context_list_projects_all_statuses_and_records_read_event(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        registry = client.app.state.main_loop.context_registry
        registry.create(
            context_id="ctx-active",
            source_session_id="session-visible",
            participant_ids=("participant-visible",),
            active_topic="topic-visible",
            parent_context_id="private-parent-sentinel",
        )
        registry.create(context_id="ctx-suspended")
        registry.suspend("ctx-suspended")
        registry.create(context_id="ctx-closed")
        registry.end("ctx-closed")

        response = client.get("/api/contexts", headers=admin_headers())

        assert response.status_code == 200
        assert [item["status"] for item in response.json()["contexts"]] == [
            "active",
            "suspended",
            "ended",
        ]
        assert set(response.json()["contexts"][0]) == {
            "context_id",
            "context_type",
            "source_channel",
            "source_session_id",
            "participant_ids",
            "active_topic",
            "active_task",
            "status",
        }
        assert "private-parent-sentinel" not in response.text

    records = EventJournal(settings.agent_journal.path).verify()
    assert any(
        record.event_type == AgentEventType.CONTEXT_READ.value
        and record.lifecycle == JournalLifecycle.COMPLETED
        for record in records
    )


def test_admin_context_list_is_empty_bounded_and_rejects_public_access(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/contexts").status_code == 200
        assert client.get("/api/contexts", headers=admin_headers()).json() == {
            "contexts": []
        }
        registry = client.app.state.main_loop.context_registry
        for index in range(3):
            registry.create(context_id=f"ctx-{index}")

        bounded = client.get(
            "/api/contexts", headers=admin_headers(), params={"limit": 2}
        )

        assert bounded.status_code == 200
        assert [item["context_id"] for item in bounded.json()["contexts"]] == [
            "ctx-1",
            "ctx-2",
        ]
        assert (
            client.get(
                "/api/contexts", headers=admin_headers(), params={"limit": 201}
            ).status_code
            == 422
        )


def test_admin_working_memory_summary_is_count_only_non_mutating_and_runtime_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        memory = client.app.state.main_loop.working_memory
        memory.admit(
            working_memory_item(
                item_id="private-item-id",
                kind=WorkingMemoryKind.CONVERSATION,
                content="raw-private-replay-sentinel",
                source_event_id="hidden-thought-sentinel",
                context_id="private-context-sentinel",
                source="secret-sentinel",
                source_channel="prompt-sentinel",
                source_session_id="private-session-sentinel",
            )
        )
        memory.admit(
            working_memory_item(
                item_id="private-reference-id",
                kind=WorkingMemoryKind.EPISODIC,
                reference="private-reference-sentinel",
            )
        )
        before = memory.items
        monkeypatch.setattr(
            memory,
            "select",
            lambda **_kwargs: pytest.fail("summary must not call mutating select"),
        )

        assert client.get("/api/state/working-memory").status_code == 200
        response = client.get("/api/state/working-memory", headers=admin_headers())

        assert response.status_code == 200
        assert response.json() == {
            "item_count": 2,
            "token_count": len("raw-private-replay-sentinel".encode("utf-8")),
            "item_capacity": memory.item_capacity,
            "token_capacity": memory.token_capacity,
        }
        assert memory.items == before
        for sentinel in (
            "raw-private-replay-sentinel",
            "private-reference-sentinel",
            "hidden-thought-sentinel",
            "secret-sentinel",
            "prompt-sentinel",
            "private-session-sentinel",
            "private-context-sentinel",
        ):
            assert sentinel not in response.text

    records = EventJournal(settings.agent_journal.path).verify()
    assert any(
        record.event_type == AgentEventType.MEMORY_READ.value
        and record.lifecycle == JournalLifecycle.COMPLETED
        and record.source == "api.state.working_memory"
        for record in records
    )


def test_admin_cockpit_outbox_summary_is_empty_bounded_and_rejects_public_access(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/outbox/summary").status_code == 200
        assert client.get("/api/outbox/summary", headers=admin_headers()).json() == {
            "pending_count": 0,
            "critical_count": 0,
            "messages": [],
        }
        outbox = client.app.state.outbox
        for index in range(3):
            outbox.enqueue(
                OutboxMessageKind.ACTION_RESULT,
                title=f"Message {index}",
                body=f"Body {index}",
                deduplication_key=f"summary-{index}",
            )

        bounded = client.get(
            "/api/outbox/summary", headers=admin_headers(), params={"limit": 2}
        )

        assert bounded.status_code == 200
        assert len(bounded.json()["messages"]) == 2
        assert (
            client.get(
                "/api/outbox/summary", headers=admin_headers(), params={"limit": 0}
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/outbox/summary", headers=admin_headers(), params={"limit": 201}
            ).status_code
            == 422
        )


def test_admin_cockpit_outbox_summary_is_safe_and_records_runtime_read(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        client.get("/api/outbox/summary", headers=admin_headers())
        outbox = client.app.state.outbox
        message = outbox.enqueue(
            OutboxMessageKind.ACTION_RESULT,
            title="Release ready",
            body="PRIVATE_SENTINEL",
            deduplication_key="PRIVATE_SENTINEL",
            interlocutor_id="PRIVATE_SENTINEL",
            references=OutboxReferences(
                event_id="event-1",
                goal_id="goal-1",
                plan_id="plan-1",
                decision_id="decision-1",
                action_id="action-1",
                commitment_id="commitment-1",
            ),
        )
        outbox.fail_delivery(message.message_id, "PRIVATE_SENTINEL")
        outbox.respond(
            message.message_id,
            kind="reply",
            actor_id="operator",
            text="PRIVATE_SENTINEL",
        )

        response = client.get("/api/outbox/summary", headers=admin_headers())

        assert response.status_code == 200
        assert response.json() == {
            "pending_count": 0,
            "critical_count": 0,
            "messages": [
                {
                    "message_id": message.message_id,
                    "title": "Action result",
                    "urgency": "normal",
                    "delivery_status": "failed",
                    "acknowledgment_status": "replied",
                    "references": {
                        "event_id": "event-1",
                        "goal_id": "goal-1",
                        "plan_id": "plan-1",
                        "decision_id": "decision-1",
                        "action_id": "action-1",
                        "commitment_id": "commitment-1",
                    },
                }
            ],
        }
        for field in (
            "body",
            "responses",
            "attempts",
            "last_failure_code",
            "deduplication_key",
            "interlocutor_id",
            "PRIVATE_SENTINEL",
        ):
            assert field not in response.text

    records = EventJournal(settings.agent_journal.path).verify()
    assert any(
        record.event_type == AgentEventType.OUTBOX_READ.value
        and record.lifecycle == JournalLifecycle.COMPLETED
        and record.source == "api.outbox.summary"
        for record in records
    )


def test_admin_cockpit_outbox_summary_counts_all_and_returns_newest_first(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        client.get("/api/outbox/summary", headers=admin_headers())
        outbox = client.app.state.outbox
        next_tick = 0

        def clock() -> datetime:
            nonlocal next_tick
            value = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=next_tick)
            next_tick += 1
            return value

        outbox.clock = clock
        created_ids: list[str] = []
        for index in range(55):
            message = outbox.enqueue(
                OutboxMessageKind.ACTION_RESULT,
                title=f"Message {index}",
                body=f"Body {index}",
                deduplication_key=f"ordered-summary-{index}",
                urgency=(OutboxUrgency.CRITICAL if index < 2 else OutboxUrgency.NORMAL),
            )
            created_ids.append(message.message_id)

        response = client.get(
            "/api/outbox/summary", headers=admin_headers(), params={"limit": 5}
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["pending_count"] == 55
        assert payload["critical_count"] == 2
        assert len(payload["messages"]) == 5
        assert [message["message_id"] for message in payload["messages"]] == list(
            reversed(created_ids[-5:])
        )


def test_outbox_public_body_preview_is_bounded_and_approval_has_no_content(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        outbox = client.app.state.outbox
        question = outbox.enqueue(
            OutboxMessageKind.QUESTION,
            title="Need a decision",
            body="Choose the local option.",
            public_preview="Choose the local option.",
            deduplication_key="preview-question",
        )
        renegotiation = outbox.enqueue(
            OutboxMessageKind.RENEGOTIATION,
            title="Constraint changed",
            body="The deadline needs to move by one day.",
            public_preview="The deadline needs to move by one day.",
            deduplication_key="preview-renegotiation",
        )
        approval = outbox.enqueue(
            OutboxMessageKind.APPROVAL_REQUEST,
            title="Approve action",
            body="The action body remains Cockpit-only.",
            deduplication_key="preview-approval",
        )

        response = client.get("/api/outbox/messages", headers=admin_headers())

        assert response.status_code == 200
        messages = {item["message_id"]: item for item in response.json()["messages"]}
        assert messages[question.message_id]["body_preview"] == question.public_preview
        assert messages[question.message_id]["body_preview"] == question.body
        assert messages[renegotiation.message_id]["body_preview"] == renegotiation.body
        assert messages[approval.message_id]["body_preview"] is None
        for item in messages.values():
            assert "body" not in item

        for kind in ("question", "renegotiation"):
            missing = client.post(
                "/api/outbox/messages",
                headers=admin_headers(),
                json={
                    "kind": kind,
                    "title": "Missing preview",
                    "body": "A message without a public preview.",
                    "deduplication_key": f"preview-missing-{kind}",
                },
            )
            assert missing.status_code == 422

        mismatched = client.post(
            "/api/outbox/messages",
            headers=admin_headers(),
            json={
                "kind": "question",
                "title": "Mismatched preview",
                "body": "Which option should continue?",
                "public_preview": "Approve an unrelated option.",
                "deduplication_key": "preview-mismatch",
            },
        )
        assert mismatched.status_code == 422

        private = client.post(
            "/api/outbox/messages",
            headers=admin_headers(),
            json={
                "kind": "question",
                "title": "Private content",
                "body": "private sentinel must not be projected",
                "public_preview": "private sentinel must not be projected",
                "deduplication_key": "preview-private",
            },
        )
        assert private.status_code == 409
        assert "private sentinel" not in private.text.lower()


def test_action_operator_summary_caps_registry_tools_independently(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        client.app.state.tool_registry = SimpleNamespace(
            list=lambda: [
                SimpleNamespace(
                    name=f"registry_tool_{index}",
                    description="Reads public metadata.",
                    tool_type=SimpleNamespace(value="metadata"),
                    status=SimpleNamespace(value="declared"),
                    generated=False,
                    human_approved=False,
                )
                for index in range(150)
            ]
        )

        response = client.get(
            "/api/actions/operator-summary?limit=1", headers=admin_headers()
        )

        assert response.status_code == 200
        assert len(response.json()["registry_tools"]) == 100
        assert (
            response.json()["registry_tools"][0]["description"]
            == "Reads public metadata."
        )
        assert (
            response.json()["registry_tools"][0]["execution_authority"]
            == "registry_only"
        )
        assert "executable" not in response.json()["registry_tools"][0]


def test_action_operator_summary_fails_closed_on_unsafe_registry_descriptions(
    tmp_path: Path,
) -> None:
    descriptions = (
        "x" * 161,
        "hidden thought: PRIVATE_SENTINEL",
        "Read /home/kagya/private/config.yaml",
        "credential token secret",
        'input_schema: {"query": "string"}',
    )
    with _client(tmp_path) as client:
        client.app.state.tool_registry = SimpleNamespace(
            list=lambda: [
                SimpleNamespace(
                    name=f"registry_tool_{index}",
                    description=description,
                    tool_type=SimpleNamespace(value="metadata"),
                    status=SimpleNamespace(value="declared"),
                    generated=False,
                    human_approved=False,
                    input_schema={"PRIVATE_SENTINEL": "secret"},
                    output_template="PRIVATE_SENTINEL",
                    metadata={"PRIVATE_SENTINEL": "secret"},
                )
                for index, description in enumerate(descriptions)
            ]
        )

        response = client.get("/api/actions/operator-summary", headers=admin_headers())

        assert response.status_code == 200
        assert [item["description"] for item in response.json()["registry_tools"]] == [
            None
        ] * len(descriptions)
        assert "PRIVATE_SENTINEL" not in response.text
        for item in response.json()["registry_tools"]:
            assert set(item) == {
                "name",
                "description",
                "tool_type",
                "status",
                "generated",
                "human_approved",
                "execution_authority",
            }


@pytest.mark.parametrize(
    "corruption",
    (
        "decision",
        "plan_id",
        "plan_revision",
        "step",
        "invalid_observation",
        "cross_bound_verification",
        "self_compensation",
        "duplicate_compensation",
        "extra_cross_bound_receipt",
    ),
)
def test_compensation_endpoint_fails_closed_on_corrupt_semantic_bindings(
    tmp_path: Path,
    corruption: str,
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "autonomy": settings.autonomy.model_copy(update={"enabled": False}),
            "outbox": settings.outbox.model_copy(
                update={"quiet_hours_start": 0, "quiet_hours_end": 0}
            ),
        }
    )
    with _client(tmp_path, settings=settings) as client:
        intent_id, command = _prepare_compensable_api_action(client, corruption)
        state = ActionState.model_validate(
            client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY]
        )
        corrupted = _corrupt_compensation_state(state, corruption)
        client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = (
            corrupted.model_dump(mode="json")
        )

        state_before = json.dumps(
            client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY],
            sort_keys=True,
        )
        decision_before = client.app.state.main_loop.decision_store.get(
            f"decision-compensation-{corruption}"
        )
        outbox_before = tuple(
            item.model_dump_json()
            for item in client.app.state.main_loop.outbox.list_messages()
        )

        summary = client.get("/api/actions/operator-summary", headers=admin_headers())

        assert summary.status_code == 200
        projected = next(
            (
                item
                for item in summary.json()["actions"]
                if item["intent_id"] == intent_id
            ),
            None,
        )
        assert projected is None or "compensate" not in projected["available_commands"]
        journal_before = len(EventJournal(settings.agent_journal.path).verify())

        rejected = client.post(
            f"/api/actions/operator/intents/{intent_id}/compensate",
            headers=admin_headers(),
            json=command,
        )

        assert rejected.status_code == 409
        assert (
            json.dumps(
                client.app.state.main_loop.persistent_state.extensions[
                    ACTION_STATE_KEY
                ],
                sort_keys=True,
            )
            == state_before
        )
        after = ActionState.model_validate(
            client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY]
        )
        assert len(after.receipts) == len(corrupted.receipts)
        assert tuple(item["status"] for item in after.notifications) == tuple(
            item["status"] for item in corrupted.notifications
        )
        assert (
            client.app.state.main_loop.decision_store.get(
                f"decision-compensation-{corruption}"
            )
            == decision_before
        )
        assert (
            tuple(
                item.model_dump_json()
                for item in client.app.state.main_loop.outbox.list_messages()
            )
            == outbox_before
        )
        assert len(EventJournal(settings.agent_journal.path).verify()) == journal_before


def test_admin_cockpit_action_trace_is_empty_private_and_runtime_ordered(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        assert client.get("/api/actions/trace").status_code == 200
        response = client.get("/api/actions/trace", headers=admin_headers())
        assert response.json() == {
            "pending_approval_count": 0,
            "retry_pending_count": 0,
            "failed_count": 0,
            "traces": [],
            "pre_intent_failures": [],
        }

    records = EventJournal(settings.agent_journal.path).verify()
    assert any(
        record.event_type == AgentEventType.ACTION_READ.value
        and record.lifecycle == JournalLifecycle.COMPLETED
        and record.source == "api.actions.trace"
        for record in records
    )


def test_admin_cockpit_action_trace_projects_related_records_without_private_data(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    pending = _cockpit_action_intent(
        "intent-pending", IntentStatus.AWAITING_APPROVAL, now, approval_id="approval-1"
    )
    succeeded = _cockpit_action_intent(
        "intent-success",
        IntentStatus.SUCCEEDED,
        now + timedelta(seconds=1),
        receipt_id="receipt-success",
    )
    failed = _cockpit_action_intent(
        "intent-failed",
        IntentStatus.FAILED,
        now + timedelta(seconds=2),
        receipt_id="receipt-timeout",
        failure_code="timeout",
    )
    compensated = _cockpit_action_intent(
        "intent-compensated",
        IntentStatus.COMPENSATED,
        now + timedelta(seconds=3),
        receipt_id="receipt-compensated",
    )
    missing = _cockpit_action_intent(
        "intent-missing",
        IntentStatus.APPROVED,
        now + timedelta(seconds=4),
        receipt_id="missing-receipt",
    )
    state = ActionState(
        intents=(pending, succeeded, failed, compensated, missing),
        approvals=(
            ApprovalRecord(
                approval_id="approval-1",
                intent_id=pending.intent_id,
                status="pending",
                requested_at=now,
                reason="PRIVATE_SENTINEL",
            ),
        ),
        receipts=(
            _cockpit_receipt(
                succeeded,
                "receipt-success",
                ReceiptStatus.SUCCEEDED,
                "observation-success",
                "verification-success",
                now,
            ),
            _cockpit_receipt(
                failed,
                "receipt-timeout",
                ReceiptStatus.TIMED_OUT,
                "observation-timeout",
                "verification-timeout",
                now,
                error_code="timeout",
            ),
            _cockpit_receipt(
                compensated,
                "receipt-original",
                ReceiptStatus.SUCCEEDED,
                None,
                None,
                now - timedelta(seconds=1),
            ),
            _cockpit_receipt(
                compensated,
                "receipt-compensated",
                ReceiptStatus.COMPENSATED,
                None,
                None,
                now,
                compensation_of="receipt-original",
            ),
        ),
        observations=(
            _cockpit_observation(
                succeeded, "receipt-success", "observation-success", True, now
            ),
            _cockpit_observation(
                failed,
                "receipt-timeout",
                "observation-timeout",
                False,
                now,
                errors=("result_fields_invalid",),
            ),
        ),
        verifications=(
            OutcomeVerification(
                verification_id="verification-success",
                intent_id=succeeded.intent_id,
                observation_id="observation-success",
                success=True,
                reason="observation_schema_valid",
                verified_at=now,
            ),
            OutcomeVerification(
                verification_id="verification-timeout",
                intent_id=failed.intent_id,
                observation_id="observation-timeout",
                success=False,
                reason="execution_timeout",
                verified_at=now,
            ),
        ),
    )
    with _client(tmp_path) as client:
        client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = (
            state.model_dump(mode="json")
        )
        before = json.dumps(
            client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY],
            sort_keys=True,
        )

        response = client.get("/api/actions/trace", headers=admin_headers())

        assert response.status_code == 200
        payload = response.json()
        assert payload["pending_approval_count"] == 1
        assert payload["retry_pending_count"] == 0
        assert payload["failed_count"] == 1
        assert [item["intent_id"] for item in payload["traces"]] == [
            "intent-missing",
            "intent-compensated",
            "intent-failed",
            "intent-success",
            "intent-pending",
        ]
        by_id = {item["intent_id"]: item for item in payload["traces"]}
        assert by_id["intent-pending"]["approval"] == {
            "approval_id": "approval-1",
            "status": "pending",
            "requested_at": now.isoformat().replace("+00:00", "Z"),
            "resolved_at": None,
            "resolved_by_operator": False,
        }
        assert by_id["intent-success"]["receipt"]["status"] == "succeeded"
        assert by_id["intent-success"]["observation"]["valid"] is True
        assert by_id["intent-success"]["verification"]["success"] is True
        assert by_id["intent-failed"]["receipt"]["status"] == "timed_out"
        assert by_id["intent-compensated"]["receipt"]["status"] == "compensated"
        assert (
            by_id["intent-compensated"]["receipt"]["compensation_of"]
            == "receipt-original"
        )
        assert by_id["intent-compensated"]["related_receipts"] == [
            {"receipt_id": "receipt-original", "status": "succeeded"}
        ]
        assert by_id["intent-missing"]["receipt"] is None
        assert by_id["intent-missing"]["observation"] is None
        assert by_id["intent-missing"]["verification"] is None
        assert "PRIVATE_SENTINEL" not in response.text
        for field in ("arguments", "preview", "idempotency_key", "data"):
            assert f'"{field}"' not in response.text
        assert (
            json.dumps(
                client.app.state.main_loop.persistent_state.extensions[
                    ACTION_STATE_KEY
                ],
                sort_keys=True,
            )
            == before
        )


def test_admin_cockpit_action_trace_counts_before_limit_and_orders_ties(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    statuses = {
        0: IntentStatus.AWAITING_APPROVAL,
        1: IntentStatus.RETRY_PENDING,
        2: IntentStatus.FAILED,
        3: IntentStatus.REJECTED,
        4: IntentStatus.CANCELLED,
    }
    intents = tuple(
        _cockpit_action_intent(
            f"intent-{index:03d}", statuses.get(index, IntentStatus.SUCCEEDED), now
        )
        for index in range(55)
    )
    with _client(tmp_path) as client:
        client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = (
            ActionState(intents=intents).model_dump(mode="json")
        )

        response = client.get(
            "/api/actions/trace", headers=admin_headers(), params={"limit": 5}
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["pending_approval_count"] == 1
        assert payload["retry_pending_count"] == 1
        assert payload["failed_count"] == 3
        assert [item["intent_id"] for item in payload["traces"]] == [
            "intent-054",
            "intent-053",
            "intent-052",
            "intent-051",
            "intent-050",
        ]
        assert (
            client.get(
                "/api/actions/trace", headers=admin_headers(), params={"limit": 0}
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/actions/trace", headers=admin_headers(), params={"limit": 201}
            ).status_code
            == 422
        )


def test_admin_cockpit_action_trace_projects_pre_intent_failures_without_duplicates(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    validation_failure = ActionValidationRecord(
        validation_id="validation-failed",
        idempotency_key="PRIVATE_SENTINEL",
        request_digest="c" * 64,
        decision_id="decision-validation",
        tool_name="document_search",
        risk_class=RiskClass.READ_ONLY,
        arguments_valid=False,
        validation_schema_revision="d" * 64,
        validation_error_codes=(ActionValidationErrorCode.ARGUMENTS_SCHEMA_INVALID,),
        validated_event_id="event-validation",
        validated_event_sequence=42,
        canonical_arguments_digest="e" * 64,
        validated_at=now,
    )
    successful_validation = ActionValidationRecord(
        validation_id="validation-policy",
        idempotency_key="PRIVATE_SENTINEL",
        request_digest="f" * 64,
        decision_id="decision-policy",
        intent_id="uncreated-intent",
        tool_name="local_notification_enqueue",
        risk_class=RiskClass.REVERSIBLE_WRITE,
        arguments_valid=True,
        validation_schema_revision="a" * 64,
        validated_event_id="event-policy",
        validated_event_sequence=43,
        canonical_arguments_digest="b" * 64,
        validated_at=now + timedelta(seconds=1),
    )
    rejection = ActionPolicyRejectionRecord(
        rejection_id="rejection-policy",
        idempotency_key="PRIVATE_SENTINEL",
        decision_id="decision-policy",
        candidate_id="candidate-policy",
        validation_id=successful_validation.validation_id,
        risk_class=RiskClass.REVERSIBLE_WRITE,
        policy_code="risk_budget_denied",
        reason_code="risk_class_exceeds_budget",
        event_id="event-policy",
        event_sequence=43,
        rejected_at=now + timedelta(seconds=1),
    )
    terminal = _cockpit_action_intent(
        "intent-terminal",
        IntentStatus.FAILED,
        now - timedelta(seconds=1),
        failure_code="timeout",
    )
    state = ActionState(
        intents=(terminal,),
        validation_records=(validation_failure, successful_validation),
        policy_rejections=(rejection,),
    )
    with _client(tmp_path) as client:
        client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = (
            state.model_dump(mode="json")
        )

        response = client.get(
            "/api/actions/trace", headers=admin_headers(), params={"limit": 1}
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["failed_count"] == 3
        assert len(payload["traces"]) == 1
        assert payload["traces"][0]["intent_id"] == terminal.intent_id
        assert len(payload["pre_intent_failures"]) == 1
        assert payload["pre_intent_failures"][0] == {
            "failure_id": "rejection-policy",
            "failure_type": "policy_rejection",
            "decision_id": "decision-policy",
            "candidate_id": "candidate-policy",
            "tool_name": "local_notification_enqueue",
            "risk_class": "reversible_write",
            "error_codes": ["risk_class_exceeds_budget"],
            "event_id": "event-policy",
            "event_sequence": 43,
            "occurred_at": "2026-07-30T00:00:01Z",
        }
        all_failures = client.get(
            "/api/actions/trace", headers=admin_headers(), params={"limit": 2}
        ).json()["pre_intent_failures"]
        assert [item["failure_id"] for item in all_failures] == [
            "rejection-policy",
            "validation-failed",
        ]
        assert all_failures[1]["decision_id"] == "decision-validation"
        assert all_failures[1]["error_codes"] == ["arguments_schema_invalid"]
        assert "PRIVATE_SENTINEL" not in response.text
        for field in (
            "idempotency_key",
            "request_digest",
            "canonical_arguments_digest",
        ):
            assert f'"{field}"' not in response.text


def test_admin_cockpit_action_trace_only_projects_allowlisted_tool_names(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    unsafe_names = (
        "PRIVATE_SENTINEL",
        "<script>alert(1)</script>",
        "tool name with spaces",
        "tool\nname",
    )
    failed_validations = tuple(
        _cockpit_validation_record(
            f"validation-unsafe-{index}",
            tool_name,
            now + timedelta(seconds=index),
            arguments_valid=False,
        )
        for index, tool_name in enumerate(unsafe_names)
    )
    allowlisted = _cockpit_validation_record(
        "validation-allowlisted",
        "document_search",
        now + timedelta(seconds=4),
        arguments_valid=False,
    )
    policy_validation = _cockpit_validation_record(
        "validation-policy-unsafe",
        "<script>alert(1)</script>",
        now + timedelta(seconds=5),
        arguments_valid=True,
    )
    rejection = ActionPolicyRejectionRecord(
        rejection_id="rejection-unsafe-tool",
        idempotency_key="PRIVATE_SENTINEL",
        decision_id="decision-policy-tool",
        candidate_id="candidate-policy-tool",
        validation_id=policy_validation.validation_id,
        risk_class=RiskClass.REVERSIBLE_WRITE,
        policy_code="risk_budget_denied",
        reason_code="risk_class_exceeds_budget",
        event_id="event-policy-tool",
        event_sequence=50,
        rejected_at=now + timedelta(seconds=5),
    )
    state = ActionState(
        validation_records=(*failed_validations, allowlisted, policy_validation),
        policy_rejections=(rejection,),
    )
    with _client(tmp_path) as client:
        client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = (
            state.model_dump(mode="json")
        )

        response = client.get("/api/actions/trace", headers=admin_headers())

        assert response.status_code == 200
        failures = {
            item["failure_id"]: item for item in response.json()["pre_intent_failures"]
        }
        for index in range(len(unsafe_names)):
            assert failures[f"validation-unsafe-{index}"]["tool_name"] is None
        assert failures["validation-allowlisted"]["tool_name"] == "document_search"
        assert failures["rejection-unsafe-tool"]["tool_name"] is None
        for value in unsafe_names:
            assert value not in response.text
        assert "<script>" not in response.text


def test_admin_cockpit_action_trace_fails_closed_on_cross_record_bindings(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    cross_decision = _cockpit_action_intent(
        "intent-cross-decision",
        IntentStatus.SUCCEEDED,
        now,
        receipt_id="receipt-cross-decision",
    )
    cross_idempotency = _cockpit_action_intent(
        "intent-cross-idempotency",
        IntentStatus.SUCCEEDED,
        now,
        receipt_id="receipt-cross-idempotency",
    )
    cross_plan = _cockpit_action_intent(
        "intent-cross-plan",
        IntentStatus.SUCCEEDED,
        now,
        receipt_id="receipt-cross-plan",
    )
    duplicate_current = _cockpit_action_intent(
        "intent-duplicate-current",
        IntentStatus.SUCCEEDED,
        now,
        receipt_id="receipt-duplicate-current",
    )
    ambiguous = _cockpit_action_intent(
        "intent-ambiguous-related",
        IntentStatus.SUCCEEDED,
        now,
        approval_id="approval-duplicate",
        receipt_id="receipt-ambiguous",
    )
    receipts = (
        _cockpit_receipt(
            cross_decision,
            "receipt-cross-decision",
            ReceiptStatus.SUCCEEDED,
            None,
            None,
            now,
        ).model_copy(update={"decision_id": "decision-other"}),
        _cockpit_receipt(
            cross_idempotency,
            "receipt-cross-idempotency",
            ReceiptStatus.SUCCEEDED,
            None,
            None,
            now,
        ).model_copy(update={"idempotency_key": "different-key"}),
        _cockpit_receipt(
            cross_plan, "receipt-cross-plan", ReceiptStatus.SUCCEEDED, None, None, now
        ).model_copy(update={"plan_revision": 2}),
        _cockpit_receipt(
            duplicate_current,
            "receipt-duplicate-current",
            ReceiptStatus.SUCCEEDED,
            None,
            None,
            now,
        ),
        _cockpit_receipt(
            duplicate_current,
            "receipt-duplicate-current",
            ReceiptStatus.FAILED,
            None,
            None,
            now,
        ),
        _cockpit_receipt(
            ambiguous,
            "receipt-ambiguous",
            ReceiptStatus.SUCCEEDED,
            "observation-duplicate",
            "verification-duplicate",
            now,
        ),
    )
    approvals = (
        ApprovalRecord(
            approval_id="approval-duplicate",
            intent_id=ambiguous.intent_id,
            status="pending",
            requested_at=now,
        ),
        ApprovalRecord(
            approval_id="approval-duplicate",
            intent_id=ambiguous.intent_id,
            status="approved",
            requested_at=now,
            resolved_at=now,
            actor_id="operator",
        ),
    )
    observations = (
        _cockpit_observation(
            ambiguous, "receipt-ambiguous", "observation-duplicate", True, now
        ),
        _cockpit_observation(
            ambiguous,
            "receipt-ambiguous",
            "observation-duplicate",
            False,
            now,
            errors=("result_fields_invalid",),
        ),
    )
    verifications = (
        OutcomeVerification(
            verification_id="verification-duplicate",
            intent_id=ambiguous.intent_id,
            observation_id="observation-duplicate",
            success=True,
            reason="observation_schema_valid",
            verified_at=now,
        ),
        OutcomeVerification(
            verification_id="verification-duplicate",
            intent_id=ambiguous.intent_id,
            observation_id="observation-duplicate",
            success=False,
            reason="observation_schema_invalid",
            verified_at=now,
        ),
    )
    state = ActionState(
        intents=(
            cross_decision,
            cross_idempotency,
            cross_plan,
            duplicate_current,
            ambiguous,
        ),
        approvals=approvals,
        receipts=receipts,
        observations=observations,
        verifications=verifications,
    )
    with _client(tmp_path) as client:
        client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = (
            state.model_dump(mode="json")
        )

        response = client.get("/api/actions/trace", headers=admin_headers())

        assert response.status_code == 200
        by_id = {item["intent_id"]: item for item in response.json()["traces"]}
        for intent_id in (
            "intent-cross-decision",
            "intent-cross-idempotency",
            "intent-cross-plan",
            "intent-duplicate-current",
        ):
            assert by_id[intent_id]["receipt"] is None
            assert by_id[intent_id]["related_receipts"] == []
        assert by_id["intent-ambiguous-related"]["receipt"] is not None
        assert by_id["intent-ambiguous-related"]["approval"] == {
            "approval_id": None,
            "status": None,
            "requested_at": None,
            "resolved_at": None,
            "resolved_by_operator": False,
        }
        assert by_id["intent-ambiguous-related"]["observation"] is None
        assert by_id["intent-ambiguous-related"]["verification"] is None


def test_admin_cockpit_action_trace_validates_compensation_bindings(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    scenario_names = ("missing", "other-intent", "other-decision", "self", "duplicate")
    intents = tuple(
        _cockpit_action_intent(
            f"intent-comp-{name}",
            IntentStatus.COMPENSATED,
            now,
            receipt_id=f"receipt-comp-{name}",
        )
        for name in scenario_names
    )
    valid = _cockpit_action_intent(
        "intent-comp-valid",
        IntentStatus.COMPENSATED,
        now,
        receipt_id="receipt-comp-valid",
    )
    by_name = {
        name: intent for name, intent in zip(scenario_names, intents, strict=True)
    }
    receipts = [
        _cockpit_receipt(
            by_name["missing"],
            "receipt-comp-missing",
            ReceiptStatus.COMPENSATED,
            None,
            None,
            now,
            compensation_of="receipt-not-found",
        ),
        _cockpit_receipt(
            by_name["other-intent"],
            "receipt-comp-other-intent",
            ReceiptStatus.COMPENSATED,
            None,
            None,
            now,
            compensation_of="receipt-target-other-intent",
        ),
        _cockpit_receipt(
            by_name["other-decision"],
            "receipt-comp-other-decision",
            ReceiptStatus.COMPENSATED,
            None,
            None,
            now,
            compensation_of="receipt-target-other-decision",
        ),
        _cockpit_receipt(
            by_name["self"],
            "receipt-comp-self",
            ReceiptStatus.COMPENSATED,
            None,
            None,
            now,
            compensation_of="receipt-comp-self",
        ),
        _cockpit_receipt(
            by_name["duplicate"],
            "receipt-comp-duplicate",
            ReceiptStatus.COMPENSATED,
            None,
            None,
            now,
            compensation_of="receipt-target-duplicate",
        ),
        _cockpit_receipt(
            valid,
            "receipt-comp-valid",
            ReceiptStatus.COMPENSATED,
            None,
            None,
            now,
            compensation_of="receipt-target-valid",
        ),
        _cockpit_receipt(
            by_name["other-intent"],
            "receipt-target-other-intent",
            ReceiptStatus.SUCCEEDED,
            None,
            None,
            now,
        ).model_copy(update={"intent_id": "different-intent"}),
        _cockpit_receipt(
            by_name["other-decision"],
            "receipt-target-other-decision",
            ReceiptStatus.SUCCEEDED,
            None,
            None,
            now,
        ).model_copy(update={"decision_id": "different-decision"}),
        _cockpit_receipt(
            by_name["duplicate"],
            "receipt-target-duplicate",
            ReceiptStatus.SUCCEEDED,
            None,
            None,
            now,
        ),
        _cockpit_receipt(
            by_name["duplicate"],
            "receipt-target-duplicate",
            ReceiptStatus.FAILED,
            None,
            None,
            now,
        ),
        _cockpit_receipt(
            valid,
            "receipt-target-valid",
            ReceiptStatus.SUCCEEDED,
            None,
            None,
            now - timedelta(seconds=1),
        ),
    ]
    state = ActionState(intents=(*intents, valid), receipts=tuple(receipts))
    with _client(tmp_path) as client:
        client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = (
            state.model_dump(mode="json")
        )

        response = client.get("/api/actions/trace", headers=admin_headers())

        assert response.status_code == 200
        by_id = {item["intent_id"]: item for item in response.json()["traces"]}
        for name in scenario_names:
            assert by_id[f"intent-comp-{name}"]["receipt"]["compensation_of"] is None
        assert (
            by_id["intent-comp-valid"]["receipt"]["compensation_of"]
            == "receipt-target-valid"
        )
        assert by_id["intent-comp-valid"]["related_receipts"] == [
            {"receipt_id": "receipt-target-valid", "status": "succeeded"}
        ]


def test_admin_cockpit_action_trace_requires_semantic_policy_binding(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    valid = _cockpit_validation_record(
        "validation-binding-valid",
        "local_notification_enqueue",
        now,
        arguments_valid=True,
    )
    decision_mismatch = _cockpit_validation_record(
        "validation-binding-decision",
        "local_notification_enqueue",
        now,
        arguments_valid=True,
    )
    risk_mismatch = _cockpit_validation_record(
        "validation-binding-risk",
        "local_notification_enqueue",
        now,
        arguments_valid=True,
    )
    invalid = _cockpit_validation_record(
        "validation-binding-invalid", "document_search", now, arguments_valid=False
    )
    duplicate_one = _cockpit_validation_record(
        "validation-binding-duplicate",
        "local_notification_enqueue",
        now,
        arguments_valid=True,
    )
    duplicate_two = duplicate_one.model_copy(update={"tool_name": "document_search"})

    def rejection(
        rejection_id: str,
        validation: ActionValidationRecord,
        *,
        decision_id: str | None = None,
        risk_class: RiskClass | None = None,
    ) -> ActionPolicyRejectionRecord:
        return ActionPolicyRejectionRecord(
            rejection_id=rejection_id,
            idempotency_key="PRIVATE_SENTINEL",
            decision_id=decision_id or validation.decision_id or "decision-missing",
            candidate_id=f"candidate-{rejection_id}",
            validation_id=validation.validation_id,
            risk_class=risk_class or validation.risk_class or RiskClass.READ_ONLY,
            policy_code="risk_budget_denied",
            reason_code="risk_class_exceeds_budget",
            event_id=f"event-{rejection_id}",
            event_sequence=51,
            rejected_at=now,
        )

    rejections = (
        rejection("rejection-binding-valid", valid),
        rejection(
            "rejection-binding-decision",
            decision_mismatch,
            decision_id="different-decision",
        ),
        rejection(
            "rejection-binding-risk", risk_mismatch, risk_class=RiskClass.READ_ONLY
        ),
        rejection("rejection-binding-invalid", invalid),
        rejection("rejection-binding-duplicate", duplicate_one),
    )
    state = ActionState(
        validation_records=(
            valid,
            decision_mismatch,
            risk_mismatch,
            invalid,
            duplicate_one,
            duplicate_two,
        ),
        policy_rejections=rejections,
    )
    with _client(tmp_path) as client:
        client.app.state.main_loop.persistent_state.extensions[ACTION_STATE_KEY] = (
            state.model_dump(mode="json")
        )

        response = client.get("/api/actions/trace", headers=admin_headers())

        assert response.status_code == 200
        failures = {
            item["failure_id"]: item for item in response.json()["pre_intent_failures"]
        }
        assert (
            failures["rejection-binding-valid"]["tool_name"]
            == "local_notification_enqueue"
        )
        for rejection_id in (
            "rejection-binding-decision",
            "rejection-binding-risk",
            "rejection-binding-invalid",
            "rejection-binding-duplicate",
        ):
            assert failures[rejection_id]["tool_name"] is None


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
    assert "Attachment metadata (untrusted data):" in prompt
    assert 'type="image"' in prompt
    assert 'name="image.png"' in prompt
    assert 'source="file"' in prompt
    assert "file:///tmp/image.png" not in prompt
    assert 'content_type="image/png"' in prompt
    assert "duration_ms" not in prompt


def test_api_chat_accepts_legacy_message_key(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/api/chat", json={"message": "hello", "attachments": []}
    )

    assert response.status_code == 200


def test_api_chat_returns_bounded_unable_for_invalid_model_output(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    client.app.state.model_provider = EmptyFallbackProvider()

    response = client.post("/api/chat", json={"text": "hello", "attachments": []})

    assert response.status_code == 200
    assert response.json()["response"] == SAFE_UNABLE_RESPONSE
    assert "primary hidden only" not in response.text
    assert "behavior_class" not in response.json()


def test_debug_chat_reports_bounded_invalid_structured_status(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    client.app.state.model_provider = EmptyFallbackProvider()

    response = client.post(
        "/api/chat/debug",
        headers=admin_headers(),
        json={"text": "hello", "attachments": [], "debug": True},
    )

    assert response.status_code == 200
    assert response.json()["response"] == SAFE_UNABLE_RESPONSE
    assert response.json()["behavior_class"] == "unable"
    assert response.json()["response_parse_valid"] is False
    assert response.json()["response_status"] == "invalid_json"


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


def test_debug_hidden_thought_requires_explicit_opt_in_and_is_never_retained(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    disabled = client.post(
        "/api/chat/debug",
        headers=admin_headers(),
        json={"text": "hello", "attachments": []},
    )
    assert disabled.status_code == 400
    assert "debug=true" in disabled.json()["detail"]

    enabled = client.post(
        "/api/chat/debug",
        headers=admin_headers(),
        json={"text": "hello", "attachments": [], "debug": True},
    )
    assert enabled.status_code == 200
    assert enabled.json()["hidden_thought"] == "debug thought"

    episode_id = enabled.json()["episode_id"]
    stored = client.app.state.memory_system.db1.get(
        ids=[episode_id], include=["documents", "metadatas"]
    )
    assert "hidden_thought" not in stored["metadatas"][0]
    assert "debug thought" not in stored["documents"][0]

    restarted = _client(tmp_path)
    restored = restarted.app.state.memory_system.get_episodic(episode_id)
    assert restored is not None
    assert not hasattr(restored, "hidden_thought")


def test_public_chat_rejects_hidden_thought_fields(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/api/chat",
        json={
            "text": "hello",
            "attachments": [],
            "hidden_thought": "must not cross the public boundary",
        },
    )

    assert response.status_code == 422
    assert "must not cross the public boundary" not in response.text


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
    }


def test_system_info_does_not_expose_secrets_or_private_paths(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/system/info")

    assert response.status_code == 200
    payload = response.text
    assert str(tmp_path) not in payload
    assert "hidden_thought" not in payload
    assert "prompt" not in payload


def test_system_events_include_fallback_without_private_fields(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.app.state.model_provider = SuccessfulFallbackProvider()

    chat = client.post("/api/chat", json={"text": "hello", "attachments": []})
    events = client.get("/api/system/events")

    assert chat.status_code == 200
    assert events.status_code == 200
    payload = events.json()
    assert payload["events"][-1]["category"] == "model"
    assert payload["events"][-1]["event_type"] == "fallback_used"
    assert payload["events"][-1]["metadata"]["model_id"] == "fallback-model"
    assert "hidden_thought" not in events.text
    assert "prompt" not in events.text


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
    settings = _settings_with_eval_set(tmp_path)
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
        json={},
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


def test_adapter_behavioral_evaluate_runs_deterministic_runtime_and_binds_valid_artifact(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    register_runtime_candidate(registry, tmp_path, "runtime-api-candidate")
    assert client.get("/api/adapters", headers=admin_headers()).status_code == 200
    before_sequence = (
        client.app.state.agent_state_store.last_snapshot.last_processed_event_sequence
    )
    before = client.app.state.agent_state_store.capture(
        client.app.state.main_loop,
        before_sequence,
    )

    response = client.post(
        "/api/adapters/runtime-api-candidate/behavioral-evaluate",
        headers=admin_headers(),
        json={
            "evaluation_id": "runtime-api-evaluation",
            "runtime_kind": "deterministic_runtime",
            "baseline_id": "base-model",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_kind"] == "deterministic_runtime"
    assert payload["artifact_status"] == "valid"
    assert payload["artifact_path"] == "behavioral/runtime-api-evaluation.json"
    entry = registry.lookup("runtime-api-candidate")
    assert entry is not None
    assert entry.behavioral_evaluation_id == "runtime-api-evaluation"
    assert entry.behavioral_artifact_state == "reconciled"
    after = client.app.state.agent_state_store.capture(
        client.app.state.main_loop,
        client.app.state.agent_state_store.last_snapshot.last_processed_event_sequence,
    )
    assert (
        after.last_processed_event_sequence == before.last_processed_event_sequence + 1
    )
    assert after.model_dump(exclude={"saved_at", "last_processed_event_sequence"}) == (
        before.model_dump(exclude={"saved_at", "last_processed_event_sequence"})
    )
    adapter_payload = client.get("/api/adapters", headers=admin_headers()).json()[
        "adapters"
    ][0]
    assert adapter_payload["behavioral_artifact_state"] == "reconciled"
    assert adapter_payload["deterministic_behavioral_artifact_status"] == "valid"
    assert adapter_payload["real_model_behavioral_artifact_status"] == "not_run"
    assert adapter_payload["behavioral_artifact_hash_match"] == "passed"
    assert not Path(adapter_payload["path"]).is_absolute()
    assert not Path(adapter_payload["behavioral_evaluation_path"]).is_absolute()

    reconciliation = client.post(
        "/api/evaluations/behavioral-reconciliation", headers=admin_headers()
    )
    assert reconciliation.status_code == 200
    assert reconciliation.json()["artifacts"][0]["status"] == "valid"


def test_adapter_evaluate_rejects_client_deterministic_scores(tmp_path: Path) -> None:
    client = _client(tmp_path)
    register_runtime_candidate(
        client.app.state.adapter_registry, tmp_path, "no-client-score"
    )

    response = client.post(
        "/api/adapters/no-client-score/evaluate",
        headers=admin_headers(),
        json={"deterministic_score": 0.99},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_production_behavioral_route_forces_real_model_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    base = _settings(tmp_path)
    monkeypatch.setenv("KAGYA_LIVE_STATE_KEY", b64encode(bytes(32)).decode("ascii"))
    settings = base.model_copy(
        update={
            "project": base.project.model_copy(
                update={"environment": ProjectEnvironment.PRODUCTION}
            ),
            "adapter_registry": base.adapter_registry.model_copy(
                update={
                    "behavioral_activation_policy": BehavioralActivationPolicy.REAL_MODEL_REQUIRED
                }
            ),
            "at_rest": base.at_rest.model_copy(
                update={
                    "live": base.at_rest.live.model_copy(
                        update={
                            "enabled": True,
                            "generation_marker": tmp_path / "sealed-generation.json",
                        }
                    ),
                    "backup": base.at_rest.backup.model_copy(
                        update={"encrypted_filesystem_attested": True}
                    ),
                    "memory_encrypted_filesystem_attested": True,
                }
            ),
        }
    )
    initialize_encrypted_state(settings)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    register_runtime_candidate(registry, tmp_path, "production-candidate")

    rejected = client.post(
        "/api/adapters/production-candidate/behavioral-evaluate",
        headers=admin_headers(),
        json={
            "evaluation_id": "production-deterministic",
            "runtime_kind": "deterministic_runtime",
        },
    )
    assert rejected.status_code == 403

    def fake_real_runner(
        settings: Settings, evaluation_id: str, **_: object
    ) -> tuple[PairedBehavioralEvaluationResult, str]:
        path = write_runtime_behavioral_result(
            registry,
            tmp_path,
            "production-candidate",
            evaluation_id=evaluation_id,
            runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
        )
        result = PairedBehavioralEvaluationResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        artifact = BehavioralArtifactStore(
            settings.adapter_registry.eval_result_dir
        ).prepare(evaluation_id, result.model_dump(mode="json"))
        return result, artifact.status.value

    monkeypatch.setattr(
        "kagya.api.routes.adapters.run_real_model_runtime_evaluation",
        fake_real_runner,
    )
    response = client.post(
        "/api/adapters/production-candidate/behavioral-evaluate",
        headers=admin_headers(),
        json={"evaluation_id": "production-real"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["runtime_kind"] == "real_model_runtime"


def test_behavioral_status_is_bounded_and_redacted(tmp_path: Path) -> None:
    client = _client(tmp_path)
    registry = client.app.state.adapter_registry
    register_runtime_candidate(registry, tmp_path, "status-candidate")

    response = client.get(
        "/api/adapters/status-candidate/behavioral-evaluation-status",
        headers=admin_headers(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "adapter_id": "status-candidate",
        "policy": "deterministic_runtime_only",
        "ordinary_gates": {
            "quality": "not_run",
            "holdout": "not_run",
            "drift": "not_run",
        },
        "deterministic_status": "not_run",
        "deterministic_coverage": "not_evaluated",
        "deterministic_artifact": "not_run",
        "real_status": "not_run",
        "real_coverage": "not_evaluated",
        "real_required": False,
        "real_artifact": "not_run",
        "activation_eligible": False,
        "activation_reason": "quality_unevaluated",
        "identity_integrity_status": "not_evaluated",
        "real_model_identity_integrity_status": "not_evaluated",
        "candidate_boundary_probe_choice": None,
        "candidate_boundary_probe_margin": None,
        "candidate_boundary_probe_count": 0,
        "rollback_reason": None,
    }
    assert str(tmp_path) not in response.text
    assert "hash" not in response.text


def test_behavioral_rerun_api_rejects_changed_runtime_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    entry = register_runtime_candidate(registry, tmp_path, "rerun-api-candidate")
    source = write_runtime_behavioral_result(
        registry,
        tmp_path,
        entry.adapter_id,
        evaluation_id="rerun-api-original",
    )
    behavioral_dir = settings.adapter_registry.eval_result_dir / "behavioral"
    behavioral_dir.mkdir(parents=True, exist_ok=True)
    source.replace(behavioral_dir / "rerun-api-original.json")
    (Path(entry.path) / "adapter_config.json").write_text(
        '{"adapter_id":"changed"}', encoding="utf-8"
    )

    response = client.post(
        "/api/evaluations/behavioral/rerun-api-original/rerun",
        headers=admin_headers(),
        json={"rerun_id": "rerun-api-replayed"},
    )

    assert response.status_code == 409
    assert "candidate_adapter_path_hash" in response.json()["detail"]


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
    assert app.state.model_provider is None


def test_adapter_endpoints_enforce_lifecycle_transitions(tmp_path: Path) -> None:
    settings = _settings_with_eval_set(tmp_path)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    candidate = register_runtime_candidate(registry, tmp_path, "adapter-api")

    invalid = client.post("/api/adapters/adapter-api/activate", headers=admin_headers())
    assert invalid.status_code == 400

    evaluated = client.post(
        "/api/adapters/adapter-api/evaluate",
        headers=admin_headers(),
        json={},
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["status"] == "trial_active"
    bind_runtime_behavioral_result(registry, tmp_path, "adapter-api")
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
    assert (
        after_activation_chat.json()["model"]["adapter_hash"] == candidate.adapter_hash
    )
    assert after_activation_chat.json()["model"]["activation_sequence"] > 0
    runtime_state = client.get("/api/adapters/runtime", headers=admin_headers())
    assert runtime_state.json()["adapter_id"] == "adapter-api"
    assert runtime_state.json()["adapter_hash"] == candidate.adapter_hash
    provenance = client.get(
        "/api/adapters/adapter-api/provenance", headers=admin_headers()
    )
    assert provenance.status_code == 200
    assert provenance.json()["adapter"]["adapter_hash"] == candidate.adapter_hash
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


def test_adapter_canary_is_fixed_server_owned_and_rolls_back_non_refusal(
    tmp_path: Path,
) -> None:
    settings = _settings_with_eval_set(tmp_path)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    register_runtime_candidate(registry, tmp_path, "adapter-canary")
    assert (
        client.post(
            "/api/adapters/adapter-canary/evaluate", headers=admin_headers(), json={}
        ).status_code
        == 200
    )
    bind_runtime_behavioral_result(registry, tmp_path, "adapter-canary")
    assert (
        client.post(
            "/api/adapters/adapter-canary/approve", headers=admin_headers()
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/adapters/adapter-canary/activate", headers=admin_headers()
        ).status_code
        == 200
    )
    provider = client.app.state.model_provider

    rejected = client.post(
        "/api/adapters/adapter-canary/canary",
        headers=admin_headers(),
        json={
            "success": True,
            "assessment_id": "caller-controlled",
            "probe_class": "refuse",
            "boundary_metadata": {"claimed_authority_ref": "authority:caller"},
        },
    )
    assert rejected.status_code == 422
    assert provider.boundary_probe_count == 0

    stable = client.post(
        "/api/adapters/adapter-canary/canary", headers=admin_headers(), json={}
    )
    assert stable.status_code == 200
    assert stable.json()["success"] is True
    assert stable.json()["automatic_rollback"] is False
    assert client.app.state.main_loop.adapter_id == "adapter-canary"

    register_runtime_candidate(registry, tmp_path, "adapter-canary-violation")
    assert (
        client.post(
            "/api/adapters/adapter-canary-violation/evaluate",
            headers=admin_headers(),
            json={},
        ).status_code
        == 200
    )
    bind_runtime_behavioral_result(registry, tmp_path, "adapter-canary-violation")
    assert (
        client.post(
            "/api/adapters/adapter-canary-violation/approve", headers=admin_headers()
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/adapters/adapter-canary-violation/activate", headers=admin_headers()
        ).status_code
        == 200
    )
    client.app.state.model_provider.boundary_probe_choice = BoundaryProbeChoice.RESPOND
    rollback = client.post(
        "/api/adapters/adapter-canary-violation/canary",
        headers=admin_headers(),
        json={},
    )
    assert rollback.status_code == 200, rollback.text
    assert rollback.json()["success"] is False
    assert rollback.json()["automatic_rollback"] is True
    assert client.app.state.main_loop.adapter_id == "adapter-canary"


def test_runtime_replacement_failure_is_atomic_and_activation_rollback_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kagya.api.dependencies as dependencies

    settings = _settings_with_eval_set(tmp_path)
    client = _client(tmp_path, settings=settings)
    registry = client.app.state.adapter_registry
    register_runtime_candidate(registry, tmp_path, "adapter-atomic")
    assert (
        client.post(
            "/api/adapters/adapter-atomic/evaluate", headers=admin_headers(), json={}
        ).status_code
        == 200
    )
    bind_runtime_behavioral_result(registry, tmp_path, "adapter-atomic")
    assert (
        client.post(
            "/api/adapters/adapter-atomic/approve", headers=admin_headers()
        ).status_code
        == 200
    )
    previous_provider = client.app.state.model_provider
    previous_loop = client.app.state.main_loop
    main_loop_type = dependencies.KagyaMainLoop
    monkeypatch.setattr(
        dependencies,
        "KagyaMainLoop",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("constructor failed")
        ),
    )

    failed_activation = client.post(
        "/api/adapters/adapter-atomic/activate", headers=admin_headers()
    )
    assert failed_activation.status_code == 400
    assert client.app.state.model_provider is previous_provider
    assert client.app.state.main_loop is previous_loop
    assert registry.lookup("adapter-atomic").status.value == "approved"

    monkeypatch.setattr(dependencies, "KagyaMainLoop", main_loop_type)
    activated = client.post(
        "/api/adapters/adapter-atomic/activate", headers=admin_headers()
    )
    assert activated.status_code == 200, activated.text
    candidate_provider = client.app.state.model_provider
    candidate_loop = client.app.state.main_loop
    monkeypatch.setattr(
        dependencies,
        "KagyaMainLoop",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("constructor failed")
        ),
    )

    failed_rollback = client.post("/api/adapters/rollback", headers=admin_headers())
    assert failed_rollback.status_code == 400
    assert client.app.state.model_provider is candidate_provider
    assert client.app.state.main_loop is candidate_loop
    assert registry.lookup("adapter-atomic").status.value == "active"

    monkeypatch.setattr(dependencies, "KagyaMainLoop", main_loop_type)
    retried = client.post("/api/adapters/rollback", headers=admin_headers())
    assert retried.status_code == 200, retried.text
    assert client.app.state.main_loop.adapter_id is None


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


def test_semantic_lifecycle_graph_api_is_idempotent_and_admin_only(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    memory = client.app.state.memory_system
    original = memory.save_semantic("original API fact")
    correction = memory.save_semantic("corrected API fact", confidence=0.8)
    payload = {
        "target_id": original,
        "relationship": "correction",
        "idempotency_key": "api-correction-1",
    }

    assert (
        client.post(
            f"/api/memory/semantic/{correction}/relationships", json=payload
        ).status_code
        == 200
    )
    related = client.post(
        f"/api/memory/semantic/{correction}/relationships",
        headers=admin_headers(),
        json=payload,
    )
    replay = client.post(
        f"/api/memory/semantic/{correction}/relationships",
        headers=admin_headers(),
        json=payload,
    )
    graph = client.get(
        f"/api/memory/semantic/{correction}/graph", headers=admin_headers()
    )
    forgotten = client.post(
        f"/api/memory/semantic/{correction}/lifecycle",
        headers=admin_headers(),
        json={"action": "forget", "idempotency_key": "api-forget-1"},
    )

    assert related.status_code == 200
    assert related.json()["supersedes_id"] == original
    assert related.json()["confidence"] == 0.8
    assert replay.json()["version"] == related.json()["version"]
    assert {item["id"] for item in graph.json()["records"]} == {original, correction}
    assert forgotten.json()["lifecycle_status"] == "forgotten"
    assert memory.retrieve_context("corrected API fact").db2_results == []


def test_agent_state_admin_snapshot_restore_and_reset(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        for response in (
            client.get("/api/state/export", headers=admin_headers()),
            client.post("/api/state/restore", headers=admin_headers(), json={}),
            client.post("/api/state/reset", headers=admin_headers()),
        ):
            assert response.status_code == 409
            assert set(response.json()) == {"detail"}
            assert set(response.json()["detail"]) == {"code"}
            assert response.json()["detail"]["code"] in {
                "private_state_projection_unavailable",
                "operator_restore_contract_required",
            }
            assert "hidden_thought" not in response.text
            assert "prompt" not in response.text

        # The authoritative Python contracts remain available to the runtime; only
        # their unsafe legacy HTTP projections are tombstoned.
        assert client.app.state.main_loop.persistent_state is not None
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

        for response in (reconstructed, dry_run):
            assert response.status_code == 409
            assert response.json() == {"detail": {"code": "raw_state_replay_disabled"}}
            assert "external_side_effects" not in response.text
            assert "hidden_thought" not in response.text
        # Internal reconstruction remains covered without exposing raw state.
        assert baseline.snapshot_hash == hash_snapshot(
            client.app.state.state_wal.reconstruct(0).snapshot
        )
        assert (
            hash_snapshot(client.app.state.agent_state_store.last_snapshot)
            == current_hash
        )

        restored = client.post("/api/state/restore/0", headers=admin_headers())
        assert restored.status_code == 409
        assert restored.json() == {"detail": {"code": "raw_state_restore_disabled"}}
        assert client.app.state.memory_system.db1.get()["ids"] == memory_ids

    records = StateWAL(settings.agent_state_wal.path).verify()
    assert all(
        record.event_type != AgentEventType.STATE_POINT_IN_TIME_RESTORE.value
        for record in records
    )


def test_governed_operator_restore_summary_preview_commit_and_retention(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        first = client.post(
            "/api/chat", json={"text": "before restore", "attachments": []}
        )
        assert first.status_code == 200
        headers = admin_headers()

        summary = client.get("/api/state/operator-restore/summary", headers=headers)
        assert summary.status_code == 200
        assert summary.json()["external_side_effects_replayed"] is False
        assert all(
            "hidden_thought" not in json.dumps(item)
            for item in summary.json()["targets"]
        )

        preview = client.post("/api/state/operator-restore/preview/0", headers=headers)
        assert preview.status_code == 200, preview.text
        preview_payload = preview.json()
        assert preview_payload["restoreable"] is True
        assert "before" not in preview_payload
        assert "after" not in preview_payload
        assert "hidden_thought" not in preview.text
        assert "prompt" not in preview.text

        commit_body = {
            "target_sequence": preview_payload["target_sequence"],
            "expected_target_hash": preview_payload["target_snapshot_hash"],
            "expected_semantic_revision": preview_payload["semantic_revision"],
            "expected_current_logical_digest": preview_payload[
                "current_logical_digest"
            ],
            "expected_preview_digest": preview_payload["preview_digest"],
            "expected_external_effect_digest": preview_payload["external_effects"][
                "effect_digest"
            ],
            "confirmation_phrase": preview_payload["confirmation_phrase"],
        }
        committed = client.post(
            "/api/state/operator-restore/commit", headers=headers, json=commit_body
        )
        assert committed.status_code == 200, committed.text
        result = committed.json()
        assert result["disposition"] == "completed"
        assert result["operation_status"] == "completed"
        assert result["external_side_effects_replayed"] is False
        assert set(result) <= {
            "command",
            "disposition",
            "operation_id",
            "event_id",
            "processing_sequence",
            "restored_target_sequence",
            "restored_target_hash",
            "post_restore_sequence",
            "post_restore_hash",
            "operation_status",
            "error_code",
            "external_side_effects_replayed",
        }
        assert "hidden_thought" not in committed.text
        assert "prompt" not in committed.text
        assert "before restore" not in committed.text

        duplicate = client.post(
            "/api/state/operator-restore/commit", headers=headers, json=commit_body
        )
        assert duplicate.status_code == 409
        assert duplicate.json() == {"detail": {"code": "restore_operation_in_progress"}}
        assert "hidden_thought" not in duplicate.text

        after = client.post(
            "/api/chat", json={"text": "after restore", "attachments": []}
        )
        assert after.status_code == 200
        assert after.json()["episode_id"] != first.json()["episode_id"]
        post_summary = client.get(
            "/api/state/operator-restore/summary", headers=headers
        )
        assert post_summary.status_code == 200
        post_payload = post_summary.json()
        assert (
            post_payload["latest_operation"]["operation_id"] == result["operation_id"]
        )
        assert post_payload["current_sequence"] == result["post_restore_sequence"] + 1
        assert (
            post_payload["retained_min_sequence"]
            <= 0
            <= post_payload["retained_max_sequence"]
        )
        assert "before restore" not in post_summary.text
        assert "after restore" not in post_summary.text

    records = EventJournal(settings.agent_journal.path).verify()
    assert any(
        record.event_type == AgentEventType.STATE_POINT_IN_TIME_RESTORE.value
        for record in records
    )
    with _client(tmp_path, settings=settings) as restarted:
        persisted = restarted.get(
            "/api/state/operator-restore/summary", headers=admin_headers()
        )
        assert persisted.status_code == 200
        assert (
            persisted.json()["latest_operation"]["operation_id"]
            == result["operation_id"]
        )
        assert persisted.json()["latest_operation"]["state"] == "completed"


def test_legacy_restore_event_does_not_block_new_governed_restore(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        outcome = client.app.state.agent_runtime.execute(
            AgentEventType.STATE_POINT_IN_TIME_RESTORE,
            source="api.state.point_in_time_restore",
            handler=lambda: None,
        )
        legacy_event_id = outcome.event.event_id

    with _client(tmp_path, settings=settings) as restarted:
        headers = admin_headers()
        summary = restarted.get(
            "/api/state/operator-restore/summary", headers=headers
        )
        assert summary.status_code == 200, summary.text
        assert summary.json()["latest_operation"] is None
        preview = restarted.post(
            "/api/state/operator-restore/preview/0", headers=headers
        )
        assert preview.status_code == 200, preview.text
        payload = preview.json()
        committed = restarted.post(
            "/api/state/operator-restore/commit",
            headers=headers,
            json={
                "target_sequence": payload["target_sequence"],
                "expected_target_hash": payload["target_snapshot_hash"],
                "expected_semantic_revision": payload["semantic_revision"],
                "expected_current_logical_digest": payload[
                    "current_logical_digest"
                ],
                "expected_preview_digest": payload["preview_digest"],
                "expected_external_effect_digest": payload["external_effects"][
                    "effect_digest"
                ],
                "confirmation_phrase": payload["confirmation_phrase"],
            },
        )
        assert committed.status_code == 200, committed.text
        latest = restarted.get(
            "/api/state/operator-restore/summary", headers=headers
        ).json()["latest_operation"]
        assert latest["operation_id"] == committed.json()["operation_id"]
        assert latest["event_id"] != legacy_event_id


def test_governed_operator_restore_rejects_exact_bindings_safely_and_standby(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        headers = admin_headers()
        missing = client.post(
            "/api/state/operator-restore/preview/999999", headers=headers
        )
        assert missing.status_code == 404
        assert missing.json() == {"detail": {"code": "restore_target_not_retained"}}
        assert "999999" not in missing.text

        preview = client.post("/api/state/operator-restore/preview/0", headers=headers)
        assert preview.status_code == 200
        payload = preview.json()
        base = {
            "target_sequence": payload["target_sequence"],
            "expected_target_hash": payload["target_snapshot_hash"],
            "expected_semantic_revision": payload["semantic_revision"],
            "expected_current_logical_digest": payload["current_logical_digest"],
            "expected_preview_digest": payload["preview_digest"],
            "expected_external_effect_digest": payload["external_effects"][
                "effect_digest"
            ],
            "confirmation_phrase": payload["confirmation_phrase"],
        }
        wrong_hash = {**base, "expected_target_hash": "a" * 64}
        rejected = client.post(
            "/api/state/operator-restore/commit", headers=headers, json=wrong_hash
        )
        assert rejected.status_code == 409
        assert rejected.json() == {"detail": {"code": "restore_preview_stale"}}
        assert "a" * 64 not in rejected.text

        client.app.state.operator_restore_service.authority = lambda _actor: False
        standby = client.post(
            "/api/state/operator-restore/commit", headers=headers, json=base
        )
        assert standby.status_code == 409
        assert standby.json() == {"detail": {"code": "restore_not_authoritative"}}
        assert "before" not in standby.text
        assert "after" not in standby.text


def test_governed_operator_restore_reports_indeterminate_without_retry(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        headers = admin_headers()
        # Persist a normal baseline before injecting a pre-WAL completion failure.
        assert (
            client.get("/api/state/working-memory", headers=headers).status_code == 200
        )
        preview = client.post(
            "/api/state/operator-restore/preview/0", headers=headers
        ).json()
        body = {
            "target_sequence": preview["target_sequence"],
            "expected_target_hash": preview["target_snapshot_hash"],
            "expected_semantic_revision": preview["semantic_revision"],
            "expected_current_logical_digest": preview["current_logical_digest"],
            "expected_preview_digest": preview["preview_digest"],
            "expected_external_effect_digest": preview["external_effects"][
                "effect_digest"
            ],
            "confirmation_phrase": preview["confirmation_phrase"],
        }

        def fail_before_wal_append(phase: str) -> None:
            if phase == "before_wal_append":
                raise OSError("PRIVATE_SENTINEL /private/state-wal")

        client.app.state.state_wal._failure_injector = fail_before_wal_append
        result = client.post(
            "/api/state/operator-restore/commit", headers=headers, json=body
        )
        assert result.status_code == 503
        assert result.json() == {"detail": {"code": "commit_indeterminate"}}
        assert "PRIVATE_SENTINEL" not in result.text
        assert "/private" not in result.text

        summary = client.get("/api/state/operator-restore/summary", headers=headers)
        assert summary.status_code == 200
        assert summary.json()["latest_operation"]["state"] == "commit_indeterminate"
        duplicate = client.post(
            "/api/state/operator-restore/commit", headers=headers, json=body
        )
        assert duplicate.status_code == 409
        assert duplicate.json() == {"detail": {"code": "restore_operation_in_progress"}}
        operation_id = summary.json()["latest_operation"]["operation_id"]

    with _client(tmp_path, settings=settings) as restarted:
        recovered = restarted.get(
            "/api/state/operator-restore/summary", headers=admin_headers()
        )
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["latest_operation"]["operation_id"] == operation_id
        assert recovered.json()["latest_operation"]["state"] == "failed"
        assert recovered.json()["latest_operation"]["error_code"] == (
            "restore_commit_failed"
        )


def test_chat_external_saga_retries_finalize_and_exposes_safe_audit(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    failures = 0

    def fail_twice(operation: str, event_id: str) -> None:
        nonlocal failures
        assert operation == "finalize"
        assert event_id
        failures += 1
        if failures < 3:
            raise RuntimeError("injected finalize failure")

    client.app.state.memory_system.set_external_failure_injector(fail_twice)
    chat = client.post(
        "/api/chat", json={"text": "saga audit private text", "attachments": []}
    )
    audit = client.get("/api/state/external-transactions", headers=admin_headers())

    assert chat.status_code == 200
    assert failures == 3
    episode = client.app.state.memory_system.get_episodic(chat.json()["episode_id"])
    assert episode is not None
    assert episode.external_transaction_status == ExternalTransactionStatus.COMMITTED
    assert audit.status_code == 409
    assert audit.json() == {"detail": {"code": "raw_state_replay_disabled"}}
    internal_audit = client.app.state.memory_system.list_external_transactions()
    assert internal_audit[0].schema_version == 1
    assert internal_audit[0].revision == 2
    assert [entry.status for entry in internal_audit[0].audit] == [
        ExternalTransactionStatus.PENDING,
        ExternalTransactionStatus.COMMITTED,
    ]
    assert "saga audit private text" not in audit.text
    assert "hidden_thought" not in audit.text


def test_internal_mutation_failure_compensates_unretrievable_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path)

    def fail_after_prepare(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected internal mutation failure")

    monkeypatch.setattr(
        client.app.state.memory_system, "link_experience", fail_after_prepare
    )
    response = client.post(
        "/api/chat", json={"text": "must never be retrieved", "attachments": []}
    )

    assert response.status_code == 500
    records = client.app.state.memory_system.list_external_transactions()
    assert len(records) == 1
    assert records[0].status == ExternalTransactionStatus.COMPENSATED
    assert [entry.status for entry in records[0].audit] == [
        ExternalTransactionStatus.PENDING,
        ExternalTransactionStatus.ORPHANED,
        ExternalTransactionStatus.COMPENSATED,
    ]
    assert (
        client.app.state.memory_system.retrieve_context(
            "must never be retrieved"
        ).db1_results
        == []
    )


def test_restart_reconciliation_finalizes_pending_chat_once(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as first:
        first.app.state.memory_system.set_external_failure_injector(
            lambda operation, event_id: (_ for _ in ()).throw(
                RuntimeError("persistent injected finalize failure")
            )
        )
        chat = first.post(
            "/api/chat", json={"text": "restart recovery", "attachments": []}
        )
        assert chat.status_code == 200
        pending = first.app.state.memory_system.get_episodic(chat.json()["episode_id"])
        assert pending is None
        transactions = first.app.state.memory_system.list_external_transactions()
        assert any(
            record.artifact_id == chat.json()["episode_id"]
            and record.status == ExternalTransactionStatus.PENDING
            for record in transactions
        )
        assert (
            first.app.state.memory_system.retrieve_context(
                "restart recovery"
            ).db1_results
            == []
        )

    with _client(tmp_path, settings=settings) as restarted:
        recovered = restarted.app.state.memory_system.get_episodic(
            chat.json()["episode_id"]
        )
        assert recovered is not None
        assert (
            recovered.external_transaction_status == ExternalTransactionStatus.COMMITTED
        )
        assert recovered.external_transaction_revision == 2
        report = restarted.app.state.external_reconciliation
        assert report.finalized == 1
        diff = restarted.get(
            "/api/state/restore/0/external-diff", headers=admin_headers()
        )
        assert diff.status_code == 409
        assert diff.json() == {"detail": {"code": "raw_state_replay_disabled"}}
        assert chat.json()["episode_id"] in {
            item.artifact_id
            for item in restarted.app.state.memory_system.list_external_transactions()
        }
        replay = restarted.post(
            "/api/state/external-transactions/reconcile", headers=admin_headers()
        )
        assert replay.status_code == 409
        assert replay.json() == {"detail": {"code": "raw_state_replay_disabled"}}
        internal_replay = (
            restarted.app.state.external_transaction_coordinator.reconcile(
                restarted.app.state.event_journal.verify()
            )
        )
        assert internal_replay.model_dump() == {
            "finalized": 0,
            "compensated": 0,
            "retryable": 0,
        }
        unchanged = restarted.app.state.memory_system.get_episodic(
            chat.json()["episode_id"]
        )
        assert unchanged is not None
        assert unchanged.external_transaction_revision == 2


def test_private_api_is_available_without_auth_headers(tmp_path: Path) -> None:
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
        == 200
    )
    assert (
        client.get("/api/memory/search", params={"query": "hello"}).status_code == 200
    )
    assert client.post("/api/sleep/jobs", json={}).status_code == 200
    assert client.get("/api/adapters").status_code == 200


def test_value_admin_lifecycle_and_structured_evaluation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()

    assert client.get("/api/values").status_code == 200
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
    assert value_state["origin_provenance"]["actor"] == "system"
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
    assert care["origin_provenance"]["actor"] == "system"
    assert care["origin_experience_ids"] == [experience_id]
    assert evidence["experience_ids"] == [experience_id]
    assert evidence["identity_origin"]["actor"] == "user"


def test_goal_and_commitment_admin_lifecycle(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()

    assert client.get("/api/goals").status_code == 200
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

    desire = client.app.state.main_loop.motivation_dynamics.observe_future_self_gap(
        "future-help",
        gap=0.8,
        uncertainty=0.2,
    )
    commitment = client.post(
        "/api/commitments",
        headers=headers,
        json={
            "commitment_id": "promise-1",
            "description": "Provide a follow-up",
            "beneficiary": "user:one",
            "scope": "Provide one follow-up response",
            "burden": 0.4,
            "interlocutor_key": "user:one",
            "related_desire_ids": [desire.motivation_id],
        },
    )
    assert commitment.status_code == 200
    assert commitment.json()["status"] == "proposed"
    assert commitment.json()["identity_origin"]["actor"] == "operator"
    assert commitment.json()["identity_origin"]["endorsement"] == "pending"
    assert (
        "promise-1" not in client.app.state.main_loop.self_model.state.commitment_refs
    )
    assert "commitment:promise-1" not in client.app.state.main_loop.goal_manager.goals
    accepted = client.post(
        "/api/commitments/promise-1/accept",
        headers=headers,
        json={
            "self_endorsement": "reviewed_scope_and_accepted",
            "value_effects": {"care": 0.5},
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "active"
    assert accepted.json()["acceptance_ref"] == "reviewed_scope_and_accepted"
    assert accepted.json()["identity_origin"]["endorsement"] == "endorsed"
    assert "promise-1" in client.app.state.main_loop.self_model.state.commitment_refs
    client.app.state.main_loop.decay_motivation(100.0)
    assert (
        client.app.state.main_loop.commitment_store.get("promise-1").status.value
        == "active"
    )

    impossible = client.post(
        "/api/commitments/promise-1/reassess",
        headers=headers,
        json={
            "fulfillability": "impossible",
            "reason": "required resource unavailable",
        },
    )
    assert impossible.status_code == 200
    decision_ref = impossible.json()["decision_refs"][-1]
    decision_id = decision_ref.removeprefix("decision:")
    decision = client.app.state.main_loop.decision_store.get(decision_id)
    assert {
        item.candidate.proposed_action for item in decision.considered_candidates
    } >= {"renegotiate_commitment", "notify_beneficiary_of_impossibility"}
    renegotiated = client.post(
        "/api/commitments/promise-1/renegotiate",
        headers=headers,
        json={
            "reason": "scope cannot be fulfilled",
            "proposed_scope": "Notify the beneficiary and agree new terms",
        },
    )
    assert renegotiated.status_code == 200
    assert renegotiated.json()["status"] == "renegotiating"
    assert renegotiated.json()["scope"] == "Provide one follow-up response"
    assert (
        renegotiated.json()["proposed_scope"]
        == "Notify the beneficiary and agree new terms"
    )
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

    breached = client.post(
        "/api/commitments",
        headers=headers,
        json={
            "commitment_id": "promise-breach",
            "description": "Send an update",
            "beneficiary": "user:one",
            "interlocutor_key": "user:one",
        },
    )
    assert breached.json()["status"] == "proposed"
    assert (
        client.post(
            "/api/commitments/promise-breach/accept",
            headers=headers,
            json={"self_endorsement": "accepted_update_responsibility"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/commitments/promise-breach/transition",
            headers=headers,
            json={"status": "breached", "reason": "update omitted"},
        ).status_code
        == 200
    )
    repaired = client.post(
        "/api/commitments/promise-breach/repair",
        headers=headers,
        json={
            "reason": "late update sent and acknowledged",
            "evidence_refs": ["experience:repair-1"],
        },
    )
    assert repaired.status_code == 200
    assert repaired.json()["accountability"][-1]["action"] == "repair"
    relationship = client.app.state.main_loop.relationship_store.for_interlocutor(
        "user:one"
    )
    assert "experience:repair-1" in relationship.repair_refs
    assert {
        item.kind
        for item in client.app.state.main_loop.narrative_self.commitment_events.values()
    } >= {"breach", "repair"}

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

    assert client.get("/api/motivation").status_code == 200
    state = client.get("/api/motivation", headers=headers)
    formed = client.post("/api/motivation/reevaluate", headers=headers)

    assert state.status_code == 200
    assert state.json()["records"]
    assert formed.status_code == 200
    assert formed.json()["goals"] == []
    _, goals = (
        client.app.state.agent_runtime.submit(
            AgentEventType.INTRINSIC_GOAL_PROPOSE,
            source="test.elapsed_motivation_review",
            handler=lambda: client.app.state.main_loop.reevaluate_motivation(
                review_at=datetime.now(UTC) + timedelta(seconds=61)
            ),
        )
        .result(timeout=2)
        .value
    )
    assert 0 < len(goals) <= 2
    goal = goals[0]
    assert goal.goal_type.value == "intrinsic"
    assert goal.identity_origin.actor.value == "self"
    assert goal.structured_target["motivation_id"]
    repeated = client.post("/api/motivation/reevaluate", headers=headers)
    assert repeated.json()["goals"] == []


def test_attention_admin_api_competes_and_controls_focus(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()
    chat = client.post(
        "/api/chat", json={"text": "private attention stimulus", "attachments": []}
    ).json()

    assert client.get("/api/attention").status_code == 200
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

    assert client.get("/api/decisions").status_code == 200
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
            "debug": True,
        },
    )
    assert belief_context.status_code == 200
    assert (
        '- Belief: "The decision context is current; '
        'status=established; confidence=0.900"' in belief_context.json()["prompt"]
    )
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
            "evidence_refs": [f"experience:{context_response['experience_id']}"],
            "goal_refs": ["decision-goal"],
            "belief_refs": ["decision-belief"],
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
    assert set(record["value_revision_refs"]) == {"honesty"}
    assert set(record["identity_origin_refs"]) == {
        "belief:decision-belief",
        "goal:decision-goal",
        "value:honesty",
    }
    assert len(record["experience_refs"]) == 1
    assert record["belief_revision_refs"] == {"decision-belief": 1}
    assert record["goal_revision_refs"] == {"decision-goal": 2}
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
    assert record["metacognition_pre_assessment_id"]
    assert record["metacognition_post_assessment_id"] is None
    explanation_response = client.post(
        "/api/decisions/decision-api-1/explanations",
        headers=headers,
        json={
            "explanation_id": "explanation-api-1",
            "context_id": context_id,
            "idempotency_key": "explanation-create-1",
        },
    )
    assert explanation_response.status_code == 200
    explanation = explanation_response.json()
    assert explanation["decision_revision"] == record["revision"]
    assert explanation["selected"]["candidate_id"] == "respond"
    assert [item["candidate_id"] for item in explanation["major_alternatives"]] == [
        "defer"
    ]
    assert {
        (item["source_type"], item["source_id"])
        for item in explanation["contributions"]
    } == {
        ("value", "honesty"),
        ("goal", "decision-goal"),
        ("belief", "decision-belief"),
    }
    assert not any(item["source_id"] == "care" for item in explanation["contributions"])
    assert explanation["outcome"]["status"] == "pending"
    assert explanation["renderer"]["state"] == "deterministic"
    assert explanation["renderer"]["visible_explanation"] == " ".join(
        {
            "disposition.selected_action.v1": "Disposition: selected action.",
            "decision_status.awaiting_outcome.v1": "Decision status: awaiting outcome.",
            "information_gaps.none.v1": "No public information gaps were recorded.",
            "outcome.pending.v1": "Outcome status: pending.",
        }[item]
        for item in explanation["renderer"]["ordered_clause_ids"]
    )
    state_before_retry = json.dumps(
        client.app.state.main_loop.persistent_state.extensions,
        sort_keys=True,
    )
    duplicate_create = client.post(
        "/api/decisions/decision-api-1/explanations",
        headers=headers,
        json={
            "explanation_id": "explanation-api-1",
            "context_id": context_id,
            "idempotency_key": "explanation-create-1",
        },
    )
    assert duplicate_create.json() == explanation
    assert (
        json.dumps(
            client.app.state.main_loop.persistent_state.extensions,
            sort_keys=True,
        )
        == state_before_retry
    )
    assert client.get("/api/decisions/explanations").status_code == 200
    filtered = client.post(
        "/api/decisions/decision-api-1/explanations",
        headers=headers,
        json={
            "explanation_id": "explanation-filtered",
            "context_id": "different-context",
            "interlocutor_id": "different-interlocutor",
            "idempotency_key": "explanation-filtered-1",
        },
    ).json()
    assert filtered["compatibility"] == "context_filtered"
    assert filtered["contributions"] == []
    assert filtered["boundary"] is None
    assert filtered["selected"]["candidate_id"] == "filtered"
    assert filtered["evidence_refs"] == []
    assert filtered["tradeoff_refs"] == []
    assert filtered["risk"]["action_intent_ref"] is None
    assert filtered["outcome"]["observed_event_ref"] is None
    assert filtered["context_id"] is None
    frame = client.app.state.main_loop.context_registry.get(context_id)
    assert frame is not None
    client.app.state.main_loop.context_registry._frames[context_id] = replace(
        frame, participant_ids=("participant-1",)
    )
    missing_interlocutor = client.post(
        "/api/decisions/decision-api-1/explanations",
        headers=headers,
        json={
            "explanation_id": "explanation-missing-interlocutor",
            "context_id": context_id,
            "idempotency_key": "explanation-missing-interlocutor-1",
        },
    ).json()
    assert missing_interlocutor["compatibility"] == "interlocutor_filtered"
    assert missing_interlocutor["contributions"] == []
    assert client.get("/api/metacognition").status_code == 200
    pre_assessment = client.get(
        f"/api/metacognition/assessments/{record['metacognition_pre_assessment_id']}",
        headers=headers,
    )
    assert pre_assessment.status_code == 200
    assert pre_assessment.json()["phase"] == "pre_decision"

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
    assert resolved.json()["metacognition_post_assessment_id"]
    revised_explanation = client.get(
        "/api/decisions/explanations/explanation-api-1", headers=headers
    ).json()
    assert revised_explanation["revision"] == 2
    assert revised_explanation["decision_revision"] == resolved.json()["revision"]
    assert revised_explanation["outcome"]["status"] == "succeeded"
    assert "outcome" in revised_explanation["change"]["changed_fields"]
    serialized_explanation = json.dumps(revised_explanation).lower()
    for sentinel in (
        "hidden_thought",
        "raw_prompt",
        "<think>",
        "credential",
        "attachment_body",
        "/home/",
    ):
        assert sentinel not in serialized_explanation
    renderer_failure = client.post(
        "/api/decisions/explanations/explanation-api-1/render",
        headers=headers,
        json={
            "expected_revision": revised_explanation["revision"],
            "idempotency_key": "explanation-render-1",
        },
    )
    assert renderer_failure.status_code == 200
    rendered = renderer_failure.json()
    assert rendered["revision"] == revised_explanation["revision"] + 1
    assert rendered["renderer"]["state"] == "failed"
    duplicate_render = client.post(
        "/api/decisions/explanations/explanation-api-1/render",
        headers=headers,
        json={
            "expected_revision": revised_explanation["revision"],
            "idempotency_key": "explanation-render-1",
        },
    )
    assert duplicate_render.json() == rendered
    assert (
        rendered["renderer"]["visible_explanation"]
        == revised_explanation["renderer"]["visible_explanation"]
    )
    metacognition = client.get("/api/metacognition", headers=headers)
    assert metacognition.status_code == 200
    assert metacognition.json()["observations"][0]["decision_id"] == "decision-api-1"
    assert "hidden_thought" not in json.dumps(metacognition.json())
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


def test_decision_explanation_survives_snapshot_wal_and_restart(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    candidate = {
        "candidate_id": "safe-no-op",
        "candidate_type": "no_op",
        "proposed_action": "safe_no_op",
        "parameters": {},
        "prerequisites": [],
        "predicted_outcomes": [],
        "uncertainty": 0.6,
        "estimated_cost": 0.0,
        "estimated_risk": 0.0,
        "value_effects": {},
        "appraisal_contributions": {},
    }
    with _client(tmp_path, settings=settings) as client:
        assert (
            client.post(
                "/api/decisions",
                headers=admin_headers(),
                json={"decision_id": "restart-decision", "candidates": [candidate]},
            ).status_code
            == 200
        )
        created = client.post(
            "/api/decisions/restart-decision/explanations",
            headers=admin_headers(),
            json={
                "explanation_id": "restart-explanation",
                "idempotency_key": "restart-explanation-create",
            },
        )
        assert created.status_code == 200
        assert created.json()["information_gap_codes"] == ["high_candidate_uncertainty"]
        failed_render = client.post(
            "/api/decisions/explanations/restart-explanation/render",
            headers=admin_headers(),
            json={
                "expected_revision": 1,
                "idempotency_key": "restart-explanation-render",
            },
        )
        assert failed_render.json()["renderer"]["state"] == "failed"

    wal_records = StateWAL(settings.agent_state_wal.path).verify()
    assert {
        "decision_explanation_create",
        "decision_explanation_render",
    }.issubset({record.event_type for record in wal_records})
    snapshot_payload = json.loads(settings.agent_state.path.read_text(encoding="utf-8"))
    persisted = snapshot_payload["extensions"]["decision_explanations"]
    assert (
        persisted["records"]["restart-explanation"][-1]["renderer"]["state"] == "failed"
    )
    for path in (
        settings.agent_state.path,
        settings.agent_state_wal.path,
        settings.agent_journal.path,
    ):
        serialized = path.read_text(encoding="utf-8").lower()
        assert "<think>" not in serialized
        assert "debug thought" not in serialized
        assert "raw_prompt" not in serialized

    with _client(tmp_path, settings=settings) as restarted:
        restored = restarted.get(
            "/api/decisions/explanations/restart-explanation",
            headers=admin_headers(),
        )
        assert restored.status_code == 200
        assert restored.json()["revision"] == 2
        assert restored.json()["renderer"]["state"] == "failed"


@pytest.mark.parametrize(
    ("candidate_type", "disposition"),
    [
        ("no_op", "no_op"),
        ("defer", "defer"),
        ("request_information", "request_information"),
    ],
)
def test_decision_explanation_dispositions_flow_through_runtime_api(
    tmp_path: Path, candidate_type: str, disposition: str
) -> None:
    with _client(tmp_path) as client:
        decision_id = f"api-{candidate_type}"
        created = client.post(
            "/api/decisions",
            headers=admin_headers(),
            json={
                "decision_id": decision_id,
                "candidates": [
                    {
                        "candidate_id": f"candidate-{candidate_type}",
                        "candidate_type": candidate_type,
                        "proposed_action": "bounded disposition",
                        "parameters": {},
                        "prerequisites": [],
                        "predicted_outcomes": [],
                        "uncertainty": 0.1,
                        "estimated_cost": 0.0,
                        "estimated_risk": 0.0,
                        "value_effects": {},
                        "appraisal_contributions": {},
                    }
                ],
            },
        )
        assert created.status_code == 200
        explanation = client.post(
            f"/api/decisions/{decision_id}/explanations",
            headers=admin_headers(),
            json={"idempotency_key": f"explain-{candidate_type}"},
        )
        assert explanation.status_code == 200
        assert explanation.json()["disposition"] == disposition


def test_self_model_evidence_revision_and_decision_integration(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = admin_headers()

    assert client.get("/api/self-model").status_code == 200
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
    assert risky_score["metacognition_contributions"] == {
        "metacognition:recommended_action": -0.75
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
        assert denied.status_code == 200
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
        assert client.get("/api/system/metrics").status_code == 200

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


def test_readiness_fails_when_memory_probe_or_reconciliation_is_unhealthy(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        memory = client.app.state.memory_system
        original_probe = memory.readiness_probe
        memory.readiness_probe = lambda: (_ for _ in ()).throw(
            RuntimeError("memory unavailable")
        )
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["shared_memory"] is False

        memory.readiness_probe = original_probe
        client.app.state.external_reconciliation = SimpleNamespace(retryable=1)
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["shared_memory"] is False


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

    assert client.get("/api/feedback").status_code == 200
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


def test_dataset_governance_browser_is_admin_only_and_supports_diff(
    tmp_path: Path,
) -> None:
    store = DatasetGovernanceStore(tmp_path / "governed-datasets")
    first = store.create_revision(
        [
            DatasetCandidate(
                input="first prompt",
                output="first response",
                provenance=DatasetProvenance("verified_episode", "episode-1"),
            )
        ]
    )
    second = store.create_revision(
        [
            DatasetCandidate(
                input="second prompt",
                output="second response",
                provenance=DatasetProvenance("verified_episode", "episode-2"),
            )
        ]
    )
    with _client(tmp_path) as client:
        client.app.state.dataset_governance = store

        assert client.get("/api/training/datasets").status_code == 200
        listing = client.get("/api/training/datasets", headers=admin_headers())
        detail = client.get(
            f"/api/training/datasets/{second.revision}", headers=admin_headers()
        )
        diff = client.get(
            "/api/training/datasets/diff",
            params={"from": first.revision, "to": second.revision},
            headers=admin_headers(),
        )

    assert listing.status_code == 200
    assert [item["revision"] for item in listing.json()["datasets"]] == [
        first.revision,
        second.revision,
    ]
    assert detail.json()["records"][0]["provenance"]["source_id"] == "episode-2"
    assert diff.status_code == 200
    assert diff.json()["from_revision"] == first.revision


def test_plan_api_is_strict_persistent_and_committed_to_wal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan_body = {
        "schema_version": 1,
        "plan_id": "api-plan",
        "goal_id": "api-plan-goal",
        "success_condition": {
            "condition_code": "verified_result",
            "required_evidence_types": ["verification"],
        },
        "failure_condition": {
            "condition_code": "failed_result",
            "required_evidence_types": ["failure"],
        },
        "abandonment_condition": {
            "condition_code": "operator_abandoned",
            "required_evidence_types": ["operator_decision"],
        },
        "steps": [
            {
                "schema_version": 1,
                "step_id": "verify",
                "action_type": "internal",
                "action_code": "verify_result",
                "parameters": {"mode": "structured"},
                "dependency_ids": [],
                "expected_observation": {
                    "observation_code": "result_verified",
                    "evidence_types": ["verification"],
                },
                "verification": {
                    "verification_code": "verify_result_evidence",
                    "required_evidence_types": ["verification"],
                    "minimum_evidence_count": 1,
                },
                "retry": {"max_attempts": 2, "backoff_seconds": 0},
                "timeout_seconds": 60,
                "rollback": {
                    "action_type": "internal",
                    "action_code": "restore_previous_state",
                    "parameters": {},
                },
            }
        ],
    }
    with _client(tmp_path, settings=settings) as client:
        assert client.get("/api/plans").status_code == 200
        assert (
            client.post(
                "/api/goals",
                headers=admin_headers(),
                json={
                    "goal_id": "api-plan-goal",
                    "goal_type": "external_request",
                    "description": "Complete a structured plan",
                },
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/goals/api-plan-goal/adopt", headers=admin_headers()
            ).status_code
            == 200
        )
        rejected = client.post(
            "/api/plans",
            headers=admin_headers(),
            json={**plan_body, "reasoning": "unstructured model prose"},
        )
        assert rejected.status_code == 422
        created = client.post("/api/plans", headers=admin_headers(), json=plan_body)
        assert created.status_code == 200
        assert (
            client.post(
                "/api/plans/api-plan/activate", headers=admin_headers()
            ).status_code
            == 200
        )
        candidates = client.get(
            "/api/plans/candidates", headers=admin_headers()
        ).json()["candidates"]
        assert candidates[0]["plan_id"] == "api-plan"
        assert candidates[0]["step_id"] == "verify"
        assert (
            client.post(
                "/api/plans/api-plan/steps/verify/start", headers=admin_headers()
            ).status_code
            == 200
        )
        no_evidence = client.post(
            "/api/plans/api-plan/steps/verify/complete",
            headers=admin_headers(),
            json={"evidence": []},
        )
        assert no_evidence.status_code == 422
        completed = client.post(
            "/api/plans/api-plan/steps/verify/complete",
            headers=admin_headers(),
            json={
                "evidence": [
                    {
                        "reference": "observation:verified-1",
                        "evidence_type": "verification",
                        "observation_code": "result_verified",
                    }
                ]
            },
        )
        assert completed.status_code == 200
        assert completed.json()["status"] == "completed"
        goal = next(
            item
            for item in client.get("/api/goals", headers=admin_headers()).json()[
                "goals"
            ]
            if item["goal_id"] == "api-plan-goal"
        )
        assert goal["status"] == "completed"

    wal_types = {
        record.event_type for record in StateWAL(settings.agent_state_wal.path).verify()
    }
    assert {"plan_update", "step_update"}.issubset(wal_types)
    with _client(tmp_path, settings=settings) as restarted:
        restored = restarted.get("/api/plans", headers=admin_headers())
        assert restored.json()["plans"][0]["status"] == "completed"
        assert restored.json()["plans"][0]["step_states"][0]["evidence"]


def _client(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
) -> TestClient:
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
            "actions": settings.actions.model_copy(
                update={
                    "document_root": tmp_path / "documents",
                    "calendar_path": tmp_path / "calendar.json",
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
        }
    )


def _settings_with_eval_set(tmp_path: Path) -> Settings:
    eval_set = tmp_path / "adapter-eval-set.json"
    eval_set.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "prompt": "adapter evaluation",
                        "expected": "DummyProvider deterministic response.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = _settings(tmp_path)
    return settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"eval_sets": [eval_set]}
            )
        }
    )


def admin_headers() -> dict[str, str]:
    return {}


def test_agency_attribution_admin_api_is_read_and_revise_only(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/attributions").status_code == 200
        listed = client.get("/api/attributions", headers=admin_headers())
        assert listed.status_code == 200
        assert listed.json() == {"attributions": []}
        assert (
            client.post(
                "/api/attributions/not-created/revisions",
                headers=admin_headers(),
                json={
                    "expected_revision": 1,
                    "contributors": [
                        {
                            "kind": "self",
                            "causal_share": 1.0,
                            "confidence": 1.0,
                            "controllability": 1.0,
                            "foreseeability": 1.0,
                            "responsibility_share": 1.0,
                        }
                    ],
                    "intended": True,
                    "uncertainty": 0.0,
                    "evidence_refs": ["observation:later"],
                    "reason_code": "later_evidence",
                },
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/attributions",
                headers=admin_headers(),
                json={},
            ).status_code
            == 405
        )


def test_counterfactual_admin_api_is_read_and_revise_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        assert client.get("/api/counterfactuals").status_code == 200
        listed = client.get("/api/counterfactuals", headers=admin_headers())
        assert listed.status_code == 200
        assert listed.json() == {"counterfactuals": []}
        assert (
            client.get(
                "/api/counterfactuals/not-created", headers=admin_headers()
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/counterfactuals",
                headers=admin_headers(),
                json={},
            ).status_code
            == 405
        )
    assert "counterfactual_read" in {
        record.event_type
        for record in EventJournal(settings.agent_journal.path).verify()
    }
    assert "counterfactual_read" in {
        record.event_type for record in StateWAL(settings.agent_state_wal.path).verify()
    }


def test_autonomy_api_persists_and_processes_operator_wakeup(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={"autonomy": settings.autonomy.model_copy(update={"enabled": False})}
    )
    with _client(tmp_path, settings=settings) as client:
        assert client.get("/api/autonomy/status").status_code == 200
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
        status_response = client.get("/api/autonomy/status", headers=admin_headers())

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


def test_started_autonomy_loop_executes_motivation_through_wal(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "autonomy": settings.autonomy.model_copy(
                update={"enabled": True, "poll_interval_seconds": 3600.0}
            )
        }
    )
    with _client(tmp_path, settings=settings) as client:
        assert client.app.state.autonomy_loop is not None
        loop = client.app.state.main_loop
        now = datetime.now(UTC)

        def persist_homeostatic_sample(observed_at: datetime) -> None:
            loop.record_homeostatic_state(
                valence=-0.7, arousal=0.8, observed_at=observed_at
            )
            loop.derive_structured_motivations()

        for seconds_ago in (120, 60):
            client.app.state.agent_runtime.submit(
                AgentEventType.EMOTION_TICK,
                source="test.persisted_homeostatic_state",
                handler=lambda seconds_ago=seconds_ago: persist_homeostatic_sample(
                    now - timedelta(seconds=seconds_ago)
                ),
            ).result(timeout=2)
        motivation = next(
            item
            for item in loop.motivation_dynamics.list_records()
            if item.source == MotivationSource.HOMEOSTATIC
        )

        cycle = client.app.state.subject_scheduler.run_cycle(
            now + timedelta(seconds=settings.autonomy.reevaluation_interval_seconds + 1)
        )

        assert cycle.inferences == 0
        assert loop.motivation_dynamics.get(motivation.motivation_id).related_goal_ids

    journal = EventJournal(settings.agent_journal.path).verify()
    wal = StateWAL(settings.agent_state_wal.path).verify()
    assert any(
        record.event_type == "autonomy_wake"
        and record.lifecycle == JournalLifecycle.COMPLETED
        for record in journal
    )
    assert any(
        record.event_type == "intrinsic_goal_propose"
        and record.lifecycle == JournalLifecycle.COMPLETED
        for record in journal
    )
    assert any(record.event_type == "autonomy_wake" for record in wal)
    assert any(record.event_type == "intrinsic_goal_propose" for record in wal)
    snapshot = json.loads(settings.agent_state.path.read_text(encoding="utf-8"))
    assert any(
        item["kind"] == "motivation_reevaluation"
        for item in snapshot["extensions"]["subject_scheduler"]["schedules"]
    )
    with _client(tmp_path, settings=settings) as restarted:
        restored = restarted.app.state.main_loop.motivation_dynamics.get(
            motivation.motivation_id
        )
        assert restored.related_goal_ids
        assert restarted.app.state.subject_scheduler.status().pending_count > 0


def test_persisted_social_and_self_model_state_rederive_motives_after_restart(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        loop = client.app.state.main_loop

        def persist_structured_sources() -> tuple[str, str]:
            loop.add_self_limitation(
                KnownLimitation(
                    limitation_id="offline-runtime",
                    description="Network access is unavailable",
                    confidence=0.9,
                    capability_ids=("research",),
                    tags=("network",),
                    evidence_refs=("policy:offline",),
                ),
                reason="persisted_runtime_constraint",
            )
            relationship = loop.relationship_store.ensure_interlocutor("person-137")
            relationship = loop.relationship_store.link_commitment(
                "person-137", "commitment:follow-up"
            )
            loop.derive_structured_motivations()
            self_motive = next(
                item
                for item in loop.motivation_dynamics.list_records()
                if item.target_ref == "limitation:offline-runtime"
            )
            social_motive = next(
                item
                for item in loop.motivation_dynamics.list_records()
                if item.target_ref == f"relationship:{relationship.relationship_id}"
            )
            return self_motive.motivation_id, social_motive.motivation_id

        self_id, social_id = (
            client.app.state.agent_runtime.submit(
                AgentEventType.INTRINSIC_GOAL_PROPOSE,
                source="test.persisted_structured_motivation_sources",
                handler=persist_structured_sources,
            )
            .result(timeout=2)
            .value
        )

    with _client(tmp_path, settings=settings) as restarted:
        records = restarted.app.state.main_loop.motivation_dynamics
        assert records.get(self_id).source == MotivationSource.LEARNING
        assert records.get(social_id).source == MotivationSource.SOCIAL
        assert records.get(self_id).evidence
        assert records.get(social_id).evidence
        assert any(
            ref.startswith("limitation:offline-runtime@")
            for ref in records.get(self_id).source_refs
        )


def test_intrinsic_proposal_is_autonomously_endorsed_planned_and_adopted(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={"autonomy": settings.autonomy.model_copy(update={"enabled": False})}
    )
    with _client(tmp_path, settings=settings) as client:
        first = client.post(
            "/api/chat",
            json={
                "text": "Bounded project signal observation alpha: build 417 completed successfully.",
                "attachments": [],
            },
        ).json()
        for text in (
            "Bounded project signal observation beta: verification batch 23 recorded three passing checks.",
            "Bounded project signal observation gamma: release candidate marker 9 is active.",
        ):
            assert (
                client.post(
                    "/api/chat",
                    json={
                        "text": text,
                        "attachments": [],
                        "context_id": first["context_id"],
                    },
                ).status_code
                == 200
            )
        loop = client.app.state.main_loop

        def prepare_and_generate() -> tuple[object, list[object]]:
            loop.working_memory.restore([])
            loop.attention_system.candidates.clear()
            loop.attention_system.compete()
            return loop.reevaluate_motivation(
                review_at=datetime.now(UTC) + timedelta(seconds=61)
            )

        _, generated = (
            client.app.state.agent_runtime.submit(
                AgentEventType.INTRINSIC_GOAL_PROPOSE,
                source="test.elapsed_intrinsic_lifecycle",
                handler=prepare_and_generate,
            )
            .result(timeout=2)
            .value
        )
        goal_id = generated[0].goal_id
        proposal = next(
            item
            for item in client.get("/api/goals", headers=admin_headers()).json()[
                "goals"
            ]
            if item["goal_id"] == goal_id
        )
        assert proposal["intrinsic_status"] == "proposal"
        assert proposal["identity_origin"]["endorsement"] == "pending"
        assert (
            client.post(
                f"/api/goals/{goal_id}/adopt", headers=admin_headers()
            ).status_code
            == 409
        )

        assert (
            client.get("/api/autonomy/status", headers=admin_headers()).status_code
            == 200
        )
        scheduler = client.app.state.subject_scheduler
        now = datetime.now(UTC) + timedelta(seconds=1)

        def set_information_gate(required: bool) -> None:
            goal = loop.goal_manager.get(goal_id)
            loop.goal_manager.goals[goal_id] = replace(goal, needs_information=required)
            loop._persist_motivation_state()

        client.app.state.agent_runtime.submit(
            AgentEventType.INTRINSIC_GOAL_DELIBERATE,
            source="test.defer_intrinsic_goal",
            handler=lambda: set_information_gate(True),
        ).result(timeout=2)
        scheduler.run_cycle(now)
        assert loop.goal_manager.get(goal_id).intrinsic_status.value == "deferred"
        client.app.state.agent_runtime.submit(
            AgentEventType.INTRINSIC_GOAL_DELIBERATE,
            source="test.release_intrinsic_goal",
            handler=lambda: set_information_gate(False),
        ).result(timeout=2)
        redeliberation_at = now + timedelta(
            seconds=settings.autonomy.reevaluation_interval_seconds + 1
        )
        scheduler.run_cycle(redeliberation_at)
        assert loop.goal_manager.get(goal_id).intrinsic_status.value == "endorsed"
        scheduler.run_cycle(redeliberation_at + timedelta(seconds=1))
        assert loop.plan_store.list_plans(goal_id=goal_id)[0].status.value == "draft"
        scheduler.run_cycle(redeliberation_at + timedelta(seconds=2))

        adopted = loop.goal_manager.get(goal_id)
        assert adopted.intrinsic_status.value == "active"
        assert adopted.status.value == "active"
        assert adopted.endorsement_provenance_refs
        assert loop.plan_store.list_plans(goal_id=goal_id)[0].status.value == "active"
        scheduler.run_cycle(redeliberation_at + timedelta(seconds=3))
        scheduler.run_cycle(redeliberation_at + timedelta(seconds=4))
        scheduler.run_cycle(redeliberation_at + timedelta(seconds=5))
        plan = loop.plan_store.list_plans(goal_id=goal_id)[0]
        assert plan.status.value == "completed"
        assert loop.goal_manager.get(goal_id).status.value == "completed"
        decision = next(
            item
            for item in loop.decision_store.list_records()
            if any(
                candidate.candidate.plan_id == plan.plan_id
                and candidate.candidate.step_id == "observe_progress"
                for candidate in item.considered_candidates
            )
        )
        selected = next(
            item.candidate
            for item in decision.considered_candidates
            if item.candidate.candidate_id == decision.selected_candidate_id
        )
        assert selected.plan_id == plan.plan_id
        assert decision.status.value == "resolved"
        action_state = client.app.state.action_execution
        intents = action_state.list_intents()
        assert len(intents) == 1
        assert intents[0].provenance.decision_id == decision.decision_id
        assert intents[0].status.value == "succeeded"
        assert len(action_state.list_receipts()) == 1
        assert len(action_state.list_observations()) == 1
        inspection = client.get("/api/goals", headers=admin_headers()).json()
        assert inspection["intrinsic_deliberations"][-1]["action"] == "endorse"
        no_goal = client.post("/api/motivation/reevaluate", headers=admin_headers())
        assert no_goal.json()["goals"] == []
        assert (
            client.get("/api/goals", headers=admin_headers()).json()[
                "intrinsic_deliberations"
            ][-1]["action"]
            == "no_goal"
        )

    snapshot = AgentStateStore(settings.agent_state.path).load(1.0)
    persisted_goal = next(
        item for item in snapshot.motivation.active_goals if item["goal_id"] == goal_id
    )
    assert persisted_goal["intrinsic_status"] == "active"
    wal = StateWAL(settings.agent_state_wal.path)
    assert wal.reconstruct().snapshot_hash == hash_snapshot(snapshot)
    event_types = {
        record.event_type
        for record in EventJournal(settings.agent_journal.path).verify()
    }
    assert event_types >= {
        "intrinsic_goal_propose",
        "intrinsic_goal_deliberate",
        "plan_generate",
        "intrinsic_goal_adopt",
        "decision_update",
        "action_intent",
        "action_execute",
    }


def test_action_api_approval_execution_receipts_and_wal_replay(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "autonomy": settings.autonomy.model_copy(update={"enabled": False}),
            "outbox": settings.outbox.model_copy(
                update={"quiet_hours_start": 0, "quiet_hours_end": 0}
            ),
        }
    )
    candidate = {
        "candidate_id": "notify",
        "candidate_type": "internal",
        "proposed_action": "Queue local notification",
        "parameters": {
            "action": {
                "tool_name": "local_notification_enqueue",
                "arguments": {
                    "channel": "local",
                    "title": "Review",
                    "body": "Please review the result",
                    "public_preview": {
                        "kind": "local_notification_enqueue",
                        "title": "Review",
                        "body": "Please review the result",
                    },
                },
            }
        },
        "prerequisites": [],
        "predicted_outcomes": [
            {
                "outcome_id": "queued",
                "description": "Notification queued",
                "probability": 1.0,
                "utility": 1.0,
            }
        ],
        "uncertainty": 0.0,
        "estimated_cost": 0.0,
        "estimated_risk": 0.1,
        "value_effects": {},
        "appraisal_contributions": {},
    }
    fallback = {
        **candidate,
        "candidate_id": "fallback",
        "candidate_type": "no_op",
        "proposed_action": "Do nothing",
        "parameters": {},
        "predicted_outcomes": [
            {
                "outcome_id": "idle",
                "description": "No action",
                "probability": 1.0,
                "utility": -1.0,
            }
        ],
        "estimated_risk": 0.0,
    }
    with _client(tmp_path, settings=settings) as client:
        assert client.get("/api/actions/intents").status_code == 409
        decision = client.post(
            "/api/decisions",
            headers=admin_headers(),
            json={
                "decision_id": "api-action-decision",
                "candidates": [candidate, fallback],
            },
        )
        assert decision.status_code == 200
        selected_explanation = client.post(
            "/api/decisions/api-action-decision/explanations",
            headers=admin_headers(),
            json={
                "explanation_id": "api-action-explanation",
                "idempotency_key": "api-action-explanation-create",
            },
        ).json()
        assert selected_explanation["disposition"] == "selected_action"
        created = client.post(
            "/api/actions/intents",
            headers=admin_headers(),
            json={
                "decision_id": "api-action-decision",
                "idempotency_key": "api-notification-1",
                "budget": {"timeout_seconds": 10.0},
            },
        )
        assert created.status_code == 200
        intent = created.json()
        assert intent["status"] == "awaiting_approval"
        assert "arguments" not in intent
        assert "idempotency_key" not in intent
        awaiting_explanation = client.post(
            "/api/decisions/explanations/api-action-explanation/revisions",
            headers=admin_headers(),
            json={
                "expected_revision": 1,
                "idempotency_key": "api-action-explanation-awaiting",
            },
        ).json()
        assert awaiting_explanation["disposition"] == "awaiting_approval"
        assert awaiting_explanation["risk"]["validation_ref"] is None
        premature_outcome = client.post(
            "/api/decisions/api-action-decision/outcome",
            headers=admin_headers(),
            json={"description": "premature", "utility": 1.0, "success": True},
        )
        assert premature_outcome.status_code == 409
        for signal in ("good", "bad"):
            feedback = client.post(
                "/api/feedback/admin",
                headers=admin_headers(),
                json={
                    "idempotency_key": f"pending-action-{signal}",
                    "feedback_id": f"pending-action-{signal}",
                    "target": {
                        "target_type": "decision",
                        "target_id": "api-action-decision",
                    },
                    "signals": [signal],
                },
            )
            assert feedback.status_code == 200
            assert (
                feedback.json()["revisions"][0]["propagation"][
                    "decision_outcome_applied"
                ]
                is False
            )
            assert (
                feedback.json()["revisions"][0]["propagation"]["value_evidence"]
                is not None
            )
            pending_decision = next(
                item
                for item in client.get(
                    "/api/decisions", headers=admin_headers()
                ).json()["decisions"]
                if item["decision_id"] == "api-action-decision"
            )
            assert pending_decision["status"] == "awaiting_outcome"
            assert pending_decision["actual_outcome"] is None
        blocked = client.post(
            f"/api/actions/intents/{intent['intent_id']}/execute",
            headers=admin_headers(),
        )
        assert blocked.status_code == 409
        assert client.get("/api/outbox/messages").status_code == 200
        approval_message = client.get(
            "/api/outbox/messages", headers=admin_headers()
        ).json()["messages"][0]
        assert approval_message["kind"] == "approval_request"
        assert approval_message["references"]["action_id"] == intent["intent_id"]
        delivered = client.post("/api/outbox/deliveries", headers=admin_headers())
        assert delivered.status_code == 200
        bypassed = client.post(
            f"/api/outbox/messages/{approval_message['message_id']}/responses",
            headers=admin_headers(),
            json={"kind": "approval", "text": "operator reviewed preview"},
        )
        assert bypassed.status_code == 409
        operator_action = client.get(
            "/api/actions/operator-summary", headers=admin_headers()
        ).json()["actions"][0]
        approved = client.post(
            f"/api/actions/operator/intents/{intent['intent_id']}/approval",
            headers=admin_headers(),
            json={
                "expected_intent_revision": operator_action["revision"],
                "expected_preview_digest": operator_action["preview"]["digest"],
                "expected_approval_id": operator_action["approval"]["approval_id"],
                "approved": True,
                "reason": "operator reviewed preview",
            },
        )
        assert approved.status_code == 200
        assert approved.json()["disposition"] == "awaiting_scheduler"
        executed = client.app.state.agent_runtime.execute(
            AgentEventType.ACTION_EXECUTE,
            source="test.scheduler.action_execution",
            handler=lambda: client.app.state.action_execution.execute(
                intent["intent_id"]
            ),
        ).value
        assert executed.status == IntentStatus.SUCCEEDED
        completed_explanation = client.get(
            "/api/decisions/explanations/api-action-explanation",
            headers=admin_headers(),
        ).json()
        assert completed_explanation["disposition"] == "selected_action"
        assert completed_explanation["outcome"]["status"] == "succeeded"
        assert completed_explanation["risk"]["receipt_ref"] == executed.receipt_id
        succeeded_revision = completed_explanation["revision"]
        assert client.get("/api/actions/receipts").status_code == 409
        trace = next(
            item
            for item in client.get("/api/actions/trace").json()["traces"]
            if item["intent_id"] == intent["intent_id"]
        )
        assert trace["receipt"]["receipt_id"] == executed.receipt_id
        assert trace["observation"] is not None
        compensable = next(
            item
            for item in client.get("/api/actions/operator-summary").json()["actions"]
            if item["intent_id"] == intent["intent_id"]
        )
        compensated = client.post(
            f"/api/actions/operator/intents/{intent['intent_id']}/compensate",
            headers=admin_headers(),
            json={
                "expected_intent_revision": compensable["revision"],
                "expected_preview_digest": compensable["preview"]["digest"],
            },
        )
        assert compensated.status_code == 200
        assert compensated.json()["action"]["status"] == "compensated"
        compensated_explanation = client.get(
            "/api/decisions/explanations/api-action-explanation",
            headers=admin_headers(),
        ).json()
        assert compensated_explanation["revision"] == succeeded_revision + 1
        assert compensated_explanation["outcome"]["status"] == "compensated"
        assert (
            compensated_explanation["risk"]["receipt_ref"]
            == compensated.json()["action"]["receipt"]["receipt_id"]
        )
        explanation_history = (
            client.app.state.main_loop.decision_explanation_store.history(
                "api-action-explanation"
            )
        )
        assert explanation_history[-2].outcome.status == "succeeded"
        assert explanation_history[-1].outcome.status == "compensated"

        assert (
            client.post(
                "/api/decisions",
                headers=admin_headers(),
                json={
                    "decision_id": "api-blocked-decision",
                    "candidates": [candidate, fallback],
                },
            ).status_code
            == 200
        )
        blocked_intent = client.post(
            "/api/actions/intents",
            headers=admin_headers(),
            json={
                "decision_id": "api-blocked-decision",
                "idempotency_key": "api-blocked-intent",
                "budget": {"max_risk_class": "read_only"},
            },
        )
        assert blocked_intent.status_code == 200, blocked_intent.text
        assert blocked_intent.json()["policy_code"] == "risk_budget_denied"
        blocked_explanation = client.post(
            "/api/decisions/api-blocked-decision/explanations",
            headers=admin_headers(),
            json={"idempotency_key": "api-blocked-explanation"},
        ).json()
        assert blocked_explanation["disposition"] == "blocked_policy"
        assert blocked_explanation["risk"]["policy_status"] == "blocked"

        assert (
            client.post(
                "/api/decisions",
                headers=admin_headers(),
                json={
                    "decision_id": "api-failed-decision",
                    "candidates": [candidate, fallback],
                },
            ).status_code
            == 200
        )
        failed_intent = client.post(
            "/api/actions/intents",
            headers=admin_headers(),
            json={
                "decision_id": "api-failed-decision",
                "idempotency_key": "api-failed-intent",
            },
        ).json()
        failed_operator_action = next(
            item
            for item in client.get(
                "/api/actions/operator-summary", headers=admin_headers()
            ).json()["actions"]
            if item["intent_id"] == failed_intent["intent_id"]
        )
        rejected = client.post(
            f"/api/actions/operator/intents/{failed_intent['intent_id']}/approval",
            headers=admin_headers(),
            json={
                "expected_intent_revision": failed_operator_action["revision"],
                "expected_preview_digest": failed_operator_action["preview"]["digest"],
                "expected_approval_id": failed_operator_action["approval"][
                    "approval_id"
                ],
                "approved": False,
                "reason": "operator_rejected",
            },
        )
        assert rejected.status_code == 200
        failed_explanation = client.post(
            "/api/decisions/api-failed-decision/explanations",
            headers=admin_headers(),
            json={"idempotency_key": "api-failed-explanation"},
        ).json()
        assert failed_explanation["disposition"] == "action_failed"
        assert failed_explanation["outcome"]["status"] == "failed"

        invalid_candidate = {
            **candidate,
            "candidate_id": "invalid-action",
            "parameters": {
                "action": {
                    "tool_name": "document_search",
                    "arguments": {"query": "x", "relative_path": "../outside"},
                }
            },
        }
        assert (
            client.post(
                "/api/decisions",
                headers=admin_headers(),
                json={
                    "decision_id": "api-invalid-decision",
                    "candidates": [invalid_candidate, fallback],
                },
            ).status_code
            == 200
        )
        invalid_result = client.post(
            "/api/actions/intents",
            headers=admin_headers(),
            json={
                "decision_id": "api-invalid-decision",
                "idempotency_key": "api-invalid-intent",
            },
        )
        assert invalid_result.status_code == 422
        action_state_after_invalid = json.dumps(
            client.app.state.main_loop.persistent_state.extensions["action_execution"],
            sort_keys=True,
        )
        duplicate_invalid = client.post(
            "/api/actions/intents",
            headers=admin_headers(),
            json={
                "decision_id": "api-invalid-decision",
                "idempotency_key": "api-invalid-intent",
            },
        )
        assert duplicate_invalid.status_code == 422
        assert duplicate_invalid.json() == invalid_result.json()
        assert (
            json.dumps(
                client.app.state.main_loop.persistent_state.extensions[
                    "action_execution"
                ],
                sort_keys=True,
            )
            == action_state_after_invalid
        )
        conflicting_invalid = client.post(
            "/api/actions/intents",
            headers=admin_headers(),
            json={
                "decision_id": "api-blocked-decision",
                "idempotency_key": "api-invalid-intent",
            },
        )
        assert conflicting_invalid.status_code == 409
        assert "different action request" in conflicting_invalid.json()["detail"]
        assert (
            json.dumps(
                client.app.state.main_loop.persistent_state.extensions[
                    "action_execution"
                ],
                sort_keys=True,
            )
            == action_state_after_invalid
        )
        invalid_explanation = client.post(
            "/api/decisions/api-invalid-decision/explanations",
            headers=admin_headers(),
            json={"idempotency_key": "api-invalid-explanation"},
        ).json()
        assert invalid_explanation["disposition"] == "unable"
        assert invalid_explanation["risk"]["policy_status"] == "invalid"

    wal_types = {
        record.event_type for record in StateWAL(settings.agent_state_wal.path).verify()
    }
    assert {"action_intent", "action_approval", "action_execute"}.issubset(wal_types)
    with _client(tmp_path, settings=settings) as restarted:
        assert restarted.get("/api/actions/intents").status_code == 409
        assert restarted.get("/api/actions/receipts").status_code == 409
        action_state = ActionState.model_validate(
            restarted.app.state.main_loop.persistent_state.extensions[
                "action_execution"
            ]
        )
        assert len(action_state.intents) == 2
        assert len(action_state.receipts) == 3
        assert any(
            item.status == IntentStatus.COMPENSATED for item in action_state.intents
        )
        assert any(
            item.status == IntentStatus.REJECTED for item in action_state.intents
        )
        outbox_messages = restarted.get(
            "/api/outbox/messages", headers=admin_headers()
        ).json()["messages"]
        assert any(item["kind"] == "action_result" for item in outbox_messages)
        assert any(
            item["kind"] == "approval_request"
            and item["acknowledgment_status"] == "approved"
            for item in outbox_messages
        )


def _cockpit_action_intent(
    intent_id: str,
    status: IntentStatus,
    timestamp: datetime,
    *,
    approval_id: str | None = None,
    receipt_id: str | None = None,
    failure_code: str | None = None,
) -> ActionIntent:
    budget = ActionBudget()
    return ActionIntent(
        intent_id=intent_id,
        revision=2,
        idempotency_key="PRIVATE_SENTINEL",
        tool_name="document_search",
        arguments={"query": "PRIVATE_SENTINEL"},
        risk_class=RiskClass.READ_ONLY,
        status=status,
        dry_run=status == IntentStatus.DRY_RUN,
        policy=PolicyEvaluation(
            evaluation_id=f"policy-{intent_id}",
            tool_name="document_search",
            risk_class=RiskClass.READ_ONLY,
            allowed=True,
            approval_required=approval_id is not None,
            reasons=("allowlisted_tool",),
            argument_digest="a" * 64,
            evaluated_at=timestamp,
        ),
        preview=ActionPreview(
            tool_name="document_search",
            risk_class=RiskClass.READ_ONLY,
            arguments={"query": "PRIVATE_SENTINEL"},
            effect="PRIVATE_SENTINEL",
            bounded_by=budget,
            compensation_available=False,
        ),
        provenance=ActionProvenance(
            decision_id="decision-1",
            candidate_id="candidate-1",
            triggering_event_id="event-1",
            plan_id="plan-1",
            plan_revision=1,
            step_id="step-1",
        ),
        budget=budget,
        attempts=1,
        cost_units_used=0,
        created_at=timestamp,
        updated_at=timestamp,
        deadline_at=timestamp + timedelta(minutes=1),
        approval_id=approval_id,
        receipt_id=receipt_id,
        failure_code=failure_code,
    )


def _cockpit_validation_record(
    validation_id: str,
    tool_name: str,
    timestamp: datetime,
    *,
    arguments_valid: bool,
) -> ActionValidationRecord:
    return ActionValidationRecord(
        validation_id=validation_id,
        idempotency_key="PRIVATE_SENTINEL",
        request_digest="c" * 64,
        decision_id=f"decision-{validation_id}",
        intent_id=f"intent-{validation_id}" if arguments_valid else None,
        tool_name=tool_name,
        risk_class=RiskClass.REVERSIBLE_WRITE if arguments_valid else None,
        arguments_valid=arguments_valid,
        validation_schema_revision="d" * 64,
        validation_error_codes=()
        if arguments_valid
        else (ActionValidationErrorCode.ARGUMENTS_SCHEMA_INVALID,),
        validated_event_id=f"event-{validation_id}",
        validated_event_sequence=45,
        canonical_arguments_digest="e" * 64,
        validated_at=timestamp,
    )


def _cockpit_receipt(
    intent: ActionIntent,
    receipt_id: str,
    status: ReceiptStatus,
    observation_id: str | None,
    verification_id: str | None,
    timestamp: datetime,
    *,
    error_code: str | None = None,
    compensation_of: str | None = None,
) -> ExecutionReceipt:
    return ExecutionReceipt(
        receipt_id=receipt_id,
        intent_id=intent.intent_id,
        idempotency_key="PRIVATE_SENTINEL",
        attempt=1,
        status=status,
        started_at=timestamp,
        finished_at=timestamp,
        duration_ms=12.5,
        observation_id=observation_id,
        verification_id=verification_id,
        event_id="event-2",
        event_sequence=45,
        decision_id="decision-1",
        plan_id="plan-1",
        plan_revision=1,
        step_id="step-1",
        compensation_of=compensation_of,
        error_code=error_code,
    )


def _cockpit_observation(
    intent: ActionIntent,
    receipt_id: str,
    observation_id: str,
    valid: bool,
    timestamp: datetime,
    *,
    errors: tuple[str, ...] = (),
) -> Observation:
    return Observation(
        observation_id=observation_id,
        intent_id=intent.intent_id,
        receipt_id=receipt_id,
        observed_at=timestamp,
        data={"result": "PRIVATE_SENTINEL"},
        result_digest="b" * 64,
        valid=valid,
        validation_errors=errors,
    )


def _prepare_compensable_api_action(
    client: TestClient, suffix: str
) -> tuple[str, dict[str, object]]:
    decision_id = f"decision-compensation-{suffix}"
    candidate = {
        "candidate_id": f"notify-{suffix}",
        "candidate_type": "internal",
        "proposed_action": "Queue local notification",
        "parameters": {
            "action": {
                "tool_name": "local_notification_enqueue",
                "arguments": {
                    "channel": "local",
                    "title": "Review",
                    "body": "Please review the result",
                    "public_preview": {
                        "kind": "local_notification_enqueue",
                        "title": "Review",
                        "body": "Please review the result",
                    },
                },
            }
        },
        "prerequisites": [],
        "predicted_outcomes": [
            {
                "outcome_id": "queued",
                "description": "Notification queued",
                "probability": 1.0,
                "utility": 1.0,
            }
        ],
        "uncertainty": 0.0,
        "estimated_cost": 0.0,
        "estimated_risk": 0.1,
        "value_effects": {},
        "appraisal_contributions": {},
    }
    fallback = {
        **candidate,
        "candidate_id": f"fallback-{suffix}",
        "candidate_type": "no_op",
        "proposed_action": "Do nothing",
        "parameters": {},
        "predicted_outcomes": [
            {
                "outcome_id": "idle",
                "description": "No action",
                "probability": 1.0,
                "utility": -1.0,
            }
        ],
        "estimated_risk": 0.0,
    }
    assert (
        client.post(
            "/api/decisions",
            headers=admin_headers(),
            json={"decision_id": decision_id, "candidates": [candidate, fallback]},
        ).status_code
        == 200
    )
    created = client.post(
        "/api/actions/intents",
        headers=admin_headers(),
        json={
            "decision_id": decision_id,
            "idempotency_key": f"compensation-{suffix}",
            "budget": {"timeout_seconds": 10.0},
        },
    )
    assert created.status_code == 200
    intent_id = created.json()["intent_id"]
    pending = next(
        item
        for item in client.get("/api/actions/operator-summary").json()["actions"]
        if item["intent_id"] == intent_id
    )
    approved = client.post(
        f"/api/actions/operator/intents/{intent_id}/approval",
        headers=admin_headers(),
        json={
            "expected_intent_revision": pending["revision"],
            "expected_preview_digest": pending["preview"]["digest"],
            "expected_approval_id": pending["approval"]["approval_id"],
            "approved": True,
        },
    )
    assert approved.status_code == 200
    executed = client.app.state.agent_runtime.execute(
        AgentEventType.ACTION_EXECUTE,
        source="test.scheduler.compensation_binding",
        handler=lambda: client.app.state.action_execution.execute(intent_id),
    ).value
    assert executed.status == IntentStatus.SUCCEEDED
    compensable = next(
        item
        for item in client.get("/api/actions/operator-summary").json()["actions"]
        if item["intent_id"] == intent_id
    )
    assert "compensate" in compensable["available_commands"]
    return intent_id, {
        "expected_intent_revision": compensable["revision"],
        "expected_preview_digest": compensable["preview"]["digest"],
    }


def _corrupt_compensation_state(state: ActionState, corruption: str) -> ActionState:
    intent = state.intents[0]
    target = next(
        item for item in state.receipts if item.receipt_id == intent.receipt_id
    )
    if corruption == "decision":
        return state.model_copy(
            update={"receipts": (target.model_copy(update={"decision_id": "other"}),)}
        )
    if corruption == "plan_id":
        return state.model_copy(
            update={"receipts": (target.model_copy(update={"plan_id": "other"}),)}
        )
    if corruption == "plan_revision":
        return state.model_copy(
            update={"receipts": (target.model_copy(update={"plan_revision": 99}),)}
        )
    if corruption == "step":
        return state.model_copy(
            update={"receipts": (target.model_copy(update={"step_id": "other"}),)}
        )
    if corruption == "invalid_observation":
        return state.model_copy(
            update={
                "observations": (
                    state.observations[0].model_copy(update={"valid": False}),
                )
            }
        )
    if corruption == "cross_bound_verification":
        other_observation = state.observations[0].model_copy(
            update={"observation_id": "observation-cross-bound"}
        )
        return state.model_copy(
            update={
                "observations": (*state.observations, other_observation),
                "verifications": (
                    state.verifications[0].model_copy(
                        update={"observation_id": other_observation.observation_id}
                    ),
                ),
            }
        )
    compensation = target.model_copy(
        update={
            "receipt_id": f"compensation-{corruption}",
            "status": ReceiptStatus.COMPENSATED,
            "compensation_of": target.receipt_id,
        }
    )
    if corruption == "self_compensation":
        compensation = compensation.model_copy(
            update={"compensation_of": compensation.receipt_id}
        )
        return state.model_copy(update={"receipts": (*state.receipts, compensation)})
    if corruption == "duplicate_compensation":
        return state.model_copy(
            update={
                "receipts": (
                    *state.receipts,
                    compensation,
                    compensation.model_copy(update={"receipt_id": "compensation-2"}),
                )
            }
        )
    if corruption == "extra_cross_bound_receipt":
        extra = target.model_copy(
            update={"receipt_id": "receipt-cross-bound", "decision_id": "other"}
        )
        return state.model_copy(update={"receipts": (*state.receipts, extra)})
    raise AssertionError(f"Unknown corruption: {corruption}")


def _wait_for_sleep_job(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/api/sleep/jobs/{job_id}", headers=admin_headers())
        data = response.json()
        if data["status"] in {"completed", "failed", "cancelled"}:
            return data
        time.sleep(0.01)
    raise AssertionError("sleep job did not finish")
