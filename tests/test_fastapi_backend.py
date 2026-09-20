import json
import math
import os
from collections.abc import Callable
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from kagya.api.server import create_app
from kagya.body import EmotionState, EmotionTemporalState
from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry
from kagya.memory import DualMemorySystem, EpisodicMemoryFormatError, MemoryContext
from kagya.memory.episodic_participant import MemoryEpisodicParticipant
from kagya.memory.working_memory_resolver import MemoryWorkingMemoryResolver
from kagya.models import DummyProvider
from kagya.persona import PromptBuilder
from kagya.runtime import (
    AbortOutcome,
    AgentEvent,
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
    AgentRuntimeStatus,
    AgentStateLoadError,
    AgentStateSaveError,
    AgentStateSaveStage,
    AgentStateSnapshotV2,
    AgentStateSnapshotV2 as AgentStateSnapshot,
    AgentStateSnapshotV3,
    AgentStateStore,
    ContextFrameSnapshot,
    ContextStateSnapshot,
    EmotionStateSnapshot,
    WorkingMemoryItemSnapshot,
    WorkingMemorySnapshot,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    EventFailureCategory,
    EventJournal,
    EventJournalAppendError,
    EventJournalAppendStage,
    EventJournalLoadError,
    EventLifecycle,
    ParticipantOutcome,
    StateWAL,
    StateWALError,
    SessionTurnParticipant,
    TransactionBinding,
    WorkingMemoryDecisionReason,
    WorkingMemory,
    WorkingMemorySourceKind,
    working_memory_item_id,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ADMIN_TOKEN = "test-admin-token"
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-R02"


class ThinkingProvider(DummyProvider):
    response_text = f"<think>{PRIVATE_SENTINEL}</think>Visible API answer."


class NonFiniteLossProvider(ThinkingProvider):
    loss_value = math.nan


class FailOnceAfterEmotionProvider(ThinkingProvider):
    def __init__(self) -> None:
        self.failed = False

    def generate(self, prompt: str) -> str:
        if not self.failed:
            self.failed = True
            raise ValueError(PRIVATE_SENTINEL)
        return super().generate(prompt)


class FailOnSecondGenerationProvider(ThinkingProvider):
    def __init__(self) -> None:
        self.generation_count = 0

    def generate(self, prompt: str) -> str:
        self.generation_count += 1
        if self.generation_count == 2:
            raise ValueError(PRIVATE_SENTINEL)
        return super().generate(prompt)


class RecordingRuntime(AgentRuntime):
    def __init__(self) -> None:
        super().__init__(queue_capacity=32)
        self.submissions: list[tuple[AgentEventType, AgentEventSource]] = []

    def submit(
        self,
        event_type: AgentEventType,
        source: AgentEventSource,
        handler: Callable[[], Any],
    ) -> Future[AgentEventOutcome[Any]]:
        self.submissions.append((event_type, source))
        return super().submit(event_type, source, handler)


class AdmissionRuntime:
    def __init__(self, error_type: type[Exception]) -> None:
        self.error_type = error_type
        self.status = AgentRuntimeStatus.CREATED

    def start(self) -> None:
        self.status = AgentRuntimeStatus.ACCEPTING

    def configure_durability(self, **_kwargs: object) -> None:
        pass

    def shutdown(self) -> None:
        self.status = AgentRuntimeStatus.STOPPED

    def submit(
        self,
        event_type: AgentEventType,
        source: AgentEventSource,
        handler: Callable[[], Any],
    ) -> NoReturn:
        event = AgentEvent("test-event", event_type, source, datetime.now(timezone.utc))
        raise self.error_type(event)


def test_health_reports_ok_after_normal_startup(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with _client(tmp_path, settings=settings) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "project": settings.project.name,
        }


def test_retention_exhaustion_blocks_mutation_but_keeps_read_only_health(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    max_bytes: int
    with _client(tmp_path, settings=settings) as client:
        journal: EventJournal = client.app.state.event_journal
        max_bytes = journal.path.stat().st_size + 1
        journal.max_bytes = max_bytes
        journal.retained_files = 2
        for message in ("fill-one", "fill-two"):
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
        saturated = journal.admission_status()
        assert not saturated.available
        assert saturated.proof_retention_blocks_safe_pruning
        before_records = journal.records
        before_high_water = journal.inspect().processing_high_water
        before_snapshot = settings.agent_state.path.read_bytes()
        before_wal = {
            path.relative_to(settings.state_wal.directory): path.read_bytes()
            for path in settings.state_wal.directory.rglob("*")
            if path.is_file()
        }
        before_memory = client.app.state.memory_system.db1.get()
        before_turns = list(client.app.state.main_loop.session_state.turns)

        response = client.post(
            "/api/chat",
            json={"message": PRIVATE_SENTINEL, "attachments": []},
        )

        assert response.status_code == 503
        assert response.json() == {
            "detail": "Agent runtime is temporarily unavailable"
        }
        assert PRIVATE_SENTINEL not in response.text
        assert str(settings.event_journal.path) not in response.text
        assert journal.records == before_records
        assert journal.inspect().processing_high_water == before_high_water
        assert settings.agent_state.path.read_bytes() == before_snapshot
        assert {
            path.relative_to(settings.state_wal.directory): path.read_bytes()
            for path in settings.state_wal.directory.rglob("*")
            if path.is_file()
        } == before_wal
        assert client.app.state.memory_system.db1.get() == before_memory
        assert client.app.state.main_loop.session_state.turns == before_turns
        assert not client.app.state.external_reconciliation_required
        assert not journal.inspect().completed_startup_reconciliations
        health = client.get("/health")
        assert health.json() == {
            "status": "degraded",
            "project": settings.project.name,
            "reason": "event_journal_retention_exhausted",
        }
        assert PRIVATE_SENTINEL not in health.text
        assert str(settings.event_journal.path) not in health.text
        read_only = client.get(
            "/api/memory/search",
            headers=admin_headers(),
            params={"query": "fill"},
        )
        assert read_only.status_code == 200

    constrained = settings.model_copy(
        update={
            "event_journal": settings.event_journal.model_copy(
                update={"max_bytes": max_bytes, "retained_files": 2}
            )
        }
    )
    with _client(tmp_path, settings=constrained) as restarted:
        assert restarted.app.state.agent_runtime.status is AgentRuntimeStatus.CREATED
        assert not restarted.app.state.external_reconciliation_required
        assert restarted.get("/health").json()["reason"] == (
            "event_journal_retention_exhausted"
        )
        assert restarted.get(
            "/api/memory/search",
            headers=admin_headers(),
            params={"query": "fill"},
        ).status_code == 200
        assert restarted.post(
            "/api/chat", json={"message": "blocked", "attachments": []}
        ).status_code == 503


def test_api_chat_works_with_dummy_provider_without_debug_leak(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        response = client.post(
            "/api/chat", json={"message": "hello", "attachments": [], "debug": False}
        )
        assert response.status_code == 200
        data = response.json()
        assert set(data) == {"episode_id", "response", "emotion", "model"}
        assert data["response"] == "Visible API answer."
        assert "hidden_thought" not in data
        assert "prompt" not in data
        assert "<think>" not in str(data)
        assert PRIVATE_SENTINEL not in str(data)
        episode = client.app.state.memory_system.get_episodic_record(
            data["episode_id"]
        )
        assert episode is not None
        assert episode.response == "Visible API answer."
        assert len(client.app.state.main_loop.session_state.turns) == 1
        records = client.app.state.event_journal.records
        assert [record.lifecycle for record in records[-8:]] == [
            EventLifecycle.ACCEPTED,
            EventLifecycle.STARTED,
            EventLifecycle.TRANSACTION_PREPARED,
            EventLifecycle.PREPARED,
            EventLifecycle.PARTICIPANT_FINALIZED,
            EventLifecycle.PARTICIPANT_FINALIZED,
            EventLifecycle.TRANSACTION_COMPLETED,
            EventLifecycle.COMPLETED,
        ]
        assert [record.participant_id for record in records[-4:-2]] == [
            "memory.episodic",
            "session.turn",
        ]
        pending = settings.memory.persist_directory / ".r07-episodic-pending"
        assert list(pending.glob("*.json")) == []


def test_direct_runtime_submit_uses_public_chat_live_authority(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        main_loop = client.app.state.main_loop
        runtime = client.app.state.agent_runtime

        ordinary = runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: main_loop.chat("direct runtime turn"),
        ).result(timeout=10)
        debug = runtime.submit(
            AgentEventType.DEBUG_CHAT,
            AgentEventSource.API_CHAT_DEBUG,
            lambda: main_loop.chat_debug("direct runtime debug turn"),
        ).result(timeout=10)

        assert ordinary.value.response == "Visible API answer."
        assert debug.value[0].response == "Visible API answer."
        assert main_loop.context_registry.current_context_id == "conversation.default"
        assert main_loop.working_memory.revision > 0
        assert len(main_loop.session_state.turns) == 2
        assert client.app.state.memory_system.get_episodic_record(
            ordinary.value.episode_id
        ) is not None
        assert client.app.state.memory_system.get_episodic_record(
            debug.value[0].episode_id
        ) is not None
        snapshot = client.app.state.agent_state_store.load()
        assert snapshot.context_state.to_registry_state() == (
            main_loop.context_registry.state
        )
        assert snapshot.working_memory.revision == main_loop.working_memory.revision
        assert (
            snapshot.emotion_state.valence,
            snapshot.emotion_state.arousal,
            snapshot.emotion_state.optimal_loss,
        ) == (
            main_loop.emotion_engine.state.valence,
            main_loop.emotion_engine.state.arousal,
            main_loop.emotion_engine.state.optimal_loss,
        )


def test_chat_selector_validation_happens_before_event_admission(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    with _client(tmp_path, runtime=runtime) as client:
        response = client.post(
            "/api/chat",
            json={
                "message": "invalid selector",
                "attachments": [],
                "context_id": "../not-an-id",
            },
        )

        assert response.status_code == 422
        assert "not-an-id" not in response.text
        assert runtime.submissions == []


def test_unrelated_validation_keeps_pre_u4_fastapi_error_shape(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/adapters/missing/evaluate",
            headers=admin_headers(),
            json={"deterministic_score": "PRIVATE-ADAPTER-INPUT"},
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail[0]["loc"] == ["body", "deterministic_score"]
        assert detail[0]["input"] == "PRIVATE-ADAPTER-INPUT"


def test_chat_session_selector_reuses_one_durable_context_without_response_metadata(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        first = client.post(
            "/api/chat",
            json={
                "message": "first session turn",
                "attachments": [],
                "client_session_id": "client-session",
                "interlocutor_key": "ref-a",
            },
        )
        second = client.post(
            "/api/chat",
            json={
                "message": "second session turn",
                "attachments": [],
                "client_session_id": "client-session",
            },
        )

        assert first.status_code == second.status_code == 200
        assert set(first.json()) == {"episode_id", "response", "emotion", "model"}
        assert set(second.json()) == {"episode_id", "response", "emotion", "model"}
        frames = client.app.state.main_loop.context_registry.state.frames
        assert len(frames) == 1
        assert frames[0].source_session_id == "client-session"
        assert frames[0].participant_refs == ("ref-a",)
        first_record = client.app.state.memory_system.get_episodic_record(
            first.json()["episode_id"]
        )
        second_record = client.app.state.memory_system.get_episodic_record(
            second.json()["episode_id"]
        )
        assert first_record is not None and second_record is not None
        assert first_record.context_id == second_record.context_id == frames[0].context_id


def test_default_context_continuity_survives_process_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as first:
        response = first.post(
            "/api/chat", json={"message": "default before restart", "attachments": []}
        )
        assert response.status_code == 200
        episode_id = response.json()["episode_id"]
        before = first.app.state.main_loop.context_registry.state
        persisted = first.app.state.agent_state_store.load().context_state
        assert tuple(frame.context_id for frame in before.frames) == (
            "conversation.default",
        )
        assert tuple(frame.context_id for frame in persisted.frames) == (
            "conversation.default",
        )
        episode = first.app.state.memory_system.get_episodic_record(episode_id)
        assert episode is not None
        assert episode.context_id == "conversation.default"

    with _client(tmp_path, settings=settings) as restarted:
        restored = restarted.app.state.main_loop.context_registry.state
        assert restored == before
        response = restarted.post(
            "/api/chat", json={"message": "default after restart", "attachments": []}
        )
        assert response.status_code == 200
        episode = restarted.app.state.memory_system.get_episodic_record(
            response.json()["episode_id"]
        )
        assert episode is not None
        assert episode.context_id == "conversation.default"
        assert restarted.app.state.main_loop.context_registry.state.frames == before.frames


def test_session_context_continuity_survives_process_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session_id = "restart-session"
    with _client(tmp_path, settings=settings) as first:
        response = first.post(
            "/api/chat",
            json={
                "message": "session before restart",
                "attachments": [],
                "client_session_id": session_id,
            },
        )
        assert response.status_code == 200
        episode = first.app.state.memory_system.get_episodic_record(
            response.json()["episode_id"]
        )
        assert episode is not None
        context_id = episode.context_id
        assert context_id is not None
        before = first.app.state.main_loop.context_registry.state
        assert [frame.source_session_id for frame in before.frames] == [session_id]

    with _client(tmp_path, settings=settings) as restarted:
        restored = restarted.app.state.main_loop.context_registry.state
        assert restored == before
        response = restarted.post(
            "/api/chat",
            json={
                "message": "session after restart",
                "attachments": [],
                "client_session_id": session_id,
            },
        )
        assert response.status_code == 200
        episode = restarted.app.state.memory_system.get_episodic_record(
            response.json()["episode_id"]
        )
        assert episode is not None
        assert episode.context_id == context_id
        frames = restarted.app.state.main_loop.context_registry.state.frames
        assert len(frames) == 1
        assert frames[0].context_id == context_id
        assert frames[0].source_session_id == session_id


def test_retained_v2_lazy_upgrade_waits_for_successful_chat(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    legacy = AgentStateSnapshotV2(
        saved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_processed_event_sequence=0,
        emotion_state=EmotionStateSnapshot(
            valence=0.1, arousal=0.2, optimal_loss=1.0
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
    )
    legacy_bytes = json.dumps(
        legacy.model_dump(mode="json"), indent=2, sort_keys=False
    ).encode()
    settings.agent_state.path.parent.mkdir(parents=True, exist_ok=True)
    settings.agent_state.path.write_bytes(legacy_bytes)

    with _client(tmp_path, settings=settings) as client:
        assert settings.agent_state.path.read_bytes() == legacy_bytes
        assert client.app.state.main_loop.context_registry.state.frames == ()
        assert not [
            record
            for record in client.app.state.event_journal.records
            if record.event_type is not None
        ]

        response = client.post(
            "/api/chat", json={"message": "upgrade context", "attachments": []}
        )
        assert response.status_code == 200
        upgraded = client.app.state.agent_state_store.load()
        assert upgraded.schema_version == 4
        assert upgraded.context_state.current_context_id == "conversation.default"
        assert tuple(
            frame.context_id for frame in upgraded.context_state.frames
        ) == ("conversation.default",)
        assert settings.agent_state.path.read_bytes() != legacy_bytes


def test_retained_v3_lazy_upgrade_waits_for_successful_chat(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    legacy_timestamp = datetime(2025, 12, 1, 12, tzinfo=timezone.utc)
    legacy_context_id = "legacy-context"
    legacy_source_id = "episode-legacy"
    legacy = AgentStateSnapshotV3(
        saved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_processed_event_sequence=0,
        emotion_state=EmotionStateSnapshot(
            valence=0.1, arousal=0.2, optimal_loss=1.0
        ),
        working_memory=WorkingMemorySnapshot(
            revision=1,
            items=(
                WorkingMemoryItemSnapshot(
                    item_id=working_memory_item_id(
                        WorkingMemorySourceKind.EPISODIC, legacy_source_id
                    ),
                    source_kind=WorkingMemorySourceKind.EPISODIC.value,
                    source_id=legacy_source_id,
                    activation=0.6,
                    salience=0.7,
                    retention_reason="recent",
                    created_revision=1,
                    last_activated_revision=1,
                ),
            ),
        ),
        context_state=ContextStateSnapshot(
            revision=1,
            current_context_id=legacy_context_id,
            frames=(
                ContextFrameSnapshot(
                    context_id=legacy_context_id,
                    context_type="conversation",
                    source_channel="chat",
                    source_session_id="legacy-session",
                    participant_refs=(),
                    parent_context_id=None,
                    related_context_ids=(),
                    status="active",
                    created_revision=1,
                    last_modified_revision=1,
                    started_at=legacy_timestamp,
                    last_active_at=legacy_timestamp,
                ),
            ),
            interlocutor_bindings=(),
        ),
    )
    legacy_bytes = json.dumps(
        legacy.model_dump(mode="json"), indent=2, sort_keys=False
    ).encode()
    settings.agent_state.path.parent.mkdir(parents=True, exist_ok=True)
    settings.agent_state.path.write_bytes(legacy_bytes)

    with _client(tmp_path, settings=settings) as client:
        assert settings.agent_state.path.read_bytes() == legacy_bytes
        assert client.app.state.agent_state_store.load() == legacy
        assert client.app.state.main_loop.loss_calibration.export() == ()
        assert client.app.state.main_loop.emotion_engine.temporal_state == (
            EmotionTemporalState()
        )
        assert client.app.state.main_loop.working_memory.revision == 1
        assert client.app.state.main_loop.working_memory.items[0].source_id == (
            legacy_source_id
        )
        assert client.app.state.main_loop.context_registry.state == (
            legacy.context_state.to_registry_state()
        )

        response = client.post(
            "/api/chat", json={"message": "upgrade v3", "attachments": []}
        )
        assert response.status_code == 200
        upgraded = client.app.state.agent_state_store.load()
        assert upgraded.schema_version == 4
        assert len(upgraded.appraisal_state.calibration_entries) == 1
        assert upgraded.appraisal_state.calibration_entries[0].count == 1
        assert (
            upgraded.appraisal_state.calibration_entries[0].mean
            == DummyProvider.loss_value
        )
        assert upgraded.appraisal_state.last_emotion_update_at is not None
        assert upgraded.working_memory.revision >= 1
        assert any(
            frame.context_id == legacy_context_id
            for frame in upgraded.context_state.frames
        )
        assert settings.agent_state.path.read_bytes() != legacy_bytes


def test_context_lifecycle_and_relation_are_durable_across_restart(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        first = client.post(
            "/api/chat",
            json={"message": "create lifecycle context", "attachments": [],
                  "client_session_id": "lifecycle-a"},
        )
        second = client.post(
            "/api/chat",
            json={"message": "create relation context", "attachments": [],
                  "client_session_id": "lifecycle-b"},
        )
        assert first.status_code == second.status_code == 200
        contexts = client.get("/api/contexts", headers=admin_headers()).json()[
            "contexts"
        ]
        context_a = next(
            item for item in contexts if item["source_session_id"] == "lifecycle-a"
        )
        context_b = next(
            item for item in contexts if item["source_session_id"] == "lifecycle-b"
        )
        context_a_id, context_b_id = context_a["context_id"], context_b["context_id"]

        suspended = client.post(
            f"/api/contexts/{context_a_id}/suspend", headers=admin_headers()
        )
        assert suspended.status_code == 200
        assert suspended.json()["status"] == "suspended"
        assert suspended.json()["is_current"] is False
        resumed = client.post(
            f"/api/contexts/{context_a_id}/resume", headers=admin_headers()
        )
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "active"
        assert resumed.json()["is_current"] is False
        related = client.post(
            f"/api/contexts/{context_a_id}/relations",
            headers=admin_headers(),
            json={"related_context_id": context_b_id},
        )
        assert related.status_code == 200
        assert context_b_id in related.json()["contexts"][0]["related_context_ids"]
        closed = client.post(
            f"/api/contexts/{context_a_id}/close", headers=admin_headers()
        )
        assert closed.status_code == 200
        assert closed.json()["status"] == "closed"
        assert closed.json()["is_current"] is False
        rejected = client.post(
            "/api/chat",
            json={"message": "closed selector", "attachments": [],
                  "context_id": context_a_id},
        )
        assert rejected.status_code == 409

    with _client(tmp_path, settings=settings) as restarted:
        restored = restarted.get("/api/contexts", headers=admin_headers())
        assert restored.status_code == 200
        by_id = {item["context_id"]: item for item in restored.json()["contexts"]}
        assert by_id[context_a_id]["status"] == "closed"
        assert by_id[context_a_id]["is_current"] is False
        assert by_id[context_b_id]["status"] == "active"
        assert context_b_id in by_id[context_a_id]["related_context_ids"]


def test_chat_context_domain_errors_are_bounded_http_responses(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        missing = client.post(
            "/api/chat",
            json={
                "message": "missing context",
                "attachments": [],
                "context_id": "missing-context",
            },
        )
        assert missing.status_code == 404
        assert missing.json() == {"detail": "Context not found"}
        assert "missing-context" not in missing.text

        selected = client.post(
            "/api/chat",
            json={
                "message": "select context",
                "attachments": [],
                "client_session_id": "client-session",
            },
        )
        assert selected.status_code == 200
        context_id = client.app.state.main_loop.context_registry.current_context_id
        assert context_id is not None
        conflict = client.post(
            "/api/chat",
            json={
                "message": "conflicting context",
                "attachments": [],
                "context_id": context_id,
                "client_session_id": "other-session",
            },
        )
        assert conflict.status_code == 409
        assert conflict.json() == {"detail": "Context selection is invalid"}
        assert "other-session" not in conflict.text


def test_api_chat_debug_requires_explicit_opt_in(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/chat/debug",
            headers=admin_headers(),
            json={"message": "hello", "attachments": [], "debug": False},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Debug access requires debug=true"


def test_api_chat_debug_is_ephemeral_and_not_persisted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        response = client.post(
            "/api/chat/debug",
            headers=admin_headers(),
            json={"message": "hello", "attachments": [], "debug": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Visible API answer."
        assert data["hidden_thought"] == PRIVATE_SENTINEL
        assert data["loss"] == DummyProvider.loss_value
        assert "prompt" in data
        assert "retrieved_memory" in data
        assert "generation_params" in data
        measurement = data["diagnostics"]["measurement"]
        assert measurement["model_key"]
        assert measurement["raw_loss"] == DummyProvider.loss_value
        assert measurement["valid"] is True
        assert measurement["invalid_reason"] is None
        assert 0.0 <= measurement["calibrated_novelty"] <= 1.0
        assert data["diagnostics"]["appraisal"]["novelty_valid"] is True
        assert all(
            data["diagnostics"]["appraisal"][field] is None
            for field in (
                "goal_progress",
                "threat",
                "controllability",
                "certainty",
                "social_relevance",
                "effort_cost",
            )
        )
        stored = client.app.state.memory_system.db1.get(
            ids=[data["episode_id"]], include=["documents", "metadatas"]
        )
        assert PRIVATE_SENTINEL not in str(stored)
        assert "hidden_thought" not in stored["metadatas"][0]
        assert PRIVATE_SENTINEL not in settings.event_journal.path.read_text()
        assert PRIVATE_SENTINEL not in settings.agent_state.path.read_text()
        assert all(
            PRIVATE_SENTINEL.encode() not in path.read_bytes()
            for path in settings.state_wal.directory.iterdir()
            if path.is_file()
        )
        assert len(client.app.state.main_loop.session_state.turns) == 1


def test_api_chat_debug_invalid_loss_is_nullable_and_not_persisted_as_zero(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(
        tmp_path, settings=settings, provider=NonFiniteLossProvider()
    ) as client:
        response = client.post(
            "/api/chat/debug",
            headers=admin_headers(),
            json={"message": "invalid loss", "attachments": [], "debug": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["loss"] is None
        assert data["diagnostics"]["measurement"]["valid"] is False
        assert data["diagnostics"]["measurement"]["raw_loss"] is None
        assert data["diagnostics"]["measurement"]["calibrated_novelty"] is None
        assert data["diagnostics"]["measurement"]["invalid_reason"] == (
            "non_finite_loss"
        )
        assert data["diagnostics"]["appraisal"]["novelty"] is None
        stored = client.app.state.memory_system.db1.get(
            ids=[data["episode_id"]], include=["documents", "metadatas"]
        )
        assert stored["metadatas"][0]["coordination_schema"] == 3
        assert stored["metadatas"][0]["loss_valid"] is False
        assert "loss" not in stored["metadatas"][0]


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
    with _client(tmp_path, settings=settings) as client:
        registry = client.app.state.adapter_registry
        registry.register_candidate(
            adapter_id="adapter-api",
            adapter_path=tmp_path / "adapter-api",
            dataset_path=tmp_path / "dataset.jsonl",
            dataset_hash="hash",
        )
        invalid = client.post(
            "/api/adapters/adapter-api/activate", headers=admin_headers()
        )
        assert invalid.status_code == 400
        evaluated = client.post(
            "/api/adapters/adapter-api/evaluate",
            headers=admin_headers(),
            json={"deterministic_score": 0.9},
        )
        assert evaluated.status_code == 200
        assert evaluated.json()["status"] == "trial_active"
        approved = client.post(
            "/api/adapters/adapter-api/approve", headers=admin_headers()
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "approved"
        active = client.post(
            "/api/adapters/adapter-api/activate", headers=admin_headers()
        )
        assert active.status_code == 200
        assert active.json()["status"] == "active"
        listed = client.get("/api/adapters", headers=admin_headers())
        assert listed.status_code == 200
        assert listed.json()["adapters"][0]["status"] == "active"


def test_sleep_endpoint_returns_dry_run_result(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        memory = client.app.state.memory_system
        memory.save_episodic(
            "sleep input",
            "sleep output",
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
        assert (
            "thought"
            not in client.app.state.settings.sleep.dream_dataset_path.read_text(
                encoding="utf-8"
            ).casefold()
        )


def test_memory_api_does_not_expose_private_fields(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        memory = client.app.state.memory_system
        episode_id = memory.save_episodic("memory input", "memory output")

        search = client.get(
            "/api/memory/search", headers=admin_headers(), params={"query": "memory"}
        )
        detail = client.get(
            f"/api/memory/episodes/{episode_id}", headers=admin_headers()
        )
        assert search.status_code == 200
        assert detail.status_code == 200
        assert "hidden_thought" not in str(search.json())
        assert "hidden_thought" not in detail.json()


def test_memory_episode_api_fails_closed_for_invalid_committed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        def invalid_record(_episode_id: str) -> None:
            raise EpisodicMemoryFormatError("PRIVATE metadata detail")

        monkeypatch.setattr(
            client.app.state.memory_system,
            "get_committed_episodic",
            invalid_record,
        )
        response = client.get(
            "/api/memory/episodes/episode-invalid", headers=admin_headers()
        )

        assert response.status_code == 500
        assert response.json() == {
            "detail": "Committed episodic Memory is invalid"
        }
        assert "PRIVATE" not in response.text


def test_sensitive_api_requires_admin_token(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert (
            client.post(
                "/api/chat",
                json={"message": "hello", "attachments": [], "debug": False},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/chat/debug",
                json={"message": "hello", "attachments": [], "debug": True},
            ).status_code
            == 401
        )
        assert (
            client.get("/api/memory/search", params={"query": "hello"}).status_code
            == 401
        )
        assert client.post("/api/sleep/run").status_code == 401
        assert client.get("/api/adapters").status_code == 401


def test_sensitive_api_reports_missing_admin_token_config(tmp_path: Path) -> None:
    with _client(tmp_path, configure_admin_token=False) as client:
        response = client.post(
            "/api/chat/debug",
            headers=admin_headers(),
            json={"message": "hello", "attachments": [], "debug": True},
        )
        assert response.status_code == 503
        assert "KAGYA_TEST_ADMIN_TOKEN" in response.json()["detail"]


def test_lifespan_owns_and_drains_one_runtime(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        runtime = client.app.state.agent_runtime
        assert runtime is client.app.state.agent_runtime
        assert runtime.status is AgentRuntimeStatus.ACCEPTING
    assert runtime.status is AgentRuntimeStatus.STOPPED


def test_mutating_routes_submit_events_without_private_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    runtime = RecordingRuntime()
    with _client(tmp_path, settings=settings, runtime=runtime) as client:
        registry = client.app.state.adapter_registry
        registry.register_candidate(
            adapter_id="recorded",
            adapter_path=tmp_path / "recorded",
            dataset_path=tmp_path / "dataset.jsonl",
            dataset_hash="hash",
        )
        assert (
            client.post(
                "/api/chat", json={"message": PRIVATE_SENTINEL, "attachments": []}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/chat/debug",
                headers=admin_headers(),
                json={"message": PRIVATE_SENTINEL, "attachments": [], "debug": True},
            ).status_code
            == 200
        )
        assert client.post("/api/sleep/run", headers=admin_headers()).status_code == 200
        assert (
            client.post(
                "/api/adapters/recorded/evaluate",
                headers=admin_headers(),
                json={"deterministic_score": 0.9},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/adapters/recorded/approve", headers=admin_headers()
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/adapters/recorded/activate", headers=admin_headers()
            ).status_code
            == 200
        )
        # Invalid transitions still admit an ADAPTER_UPDATE event before the domain 400.
        assert (
            client.post(
                "/api/adapters/missing/trial", headers=admin_headers()
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/adapters/missing/reject", headers=admin_headers()
            ).status_code
            == 400
        )

    assert [(kind, source) for kind, source in runtime.submissions] == [
        (AgentEventType.CHAT, AgentEventSource.API_CHAT),
        (AgentEventType.DEBUG_CHAT, AgentEventSource.API_CHAT_DEBUG),
        (AgentEventType.SLEEP, AgentEventSource.API_SLEEP_RUN),
        (AgentEventType.ADAPTER_EVALUATE, AgentEventSource.API_ADAPTER_EVALUATE),
        (AgentEventType.ADAPTER_UPDATE, AgentEventSource.API_ADAPTER_APPROVE),
        (AgentEventType.ADAPTER_UPDATE, AgentEventSource.API_ADAPTER_ACTIVATE),
        (AgentEventType.ADAPTER_UPDATE, AgentEventSource.API_ADAPTER_TRIAL),
        (AgentEventType.ADAPTER_UPDATE, AgentEventSource.API_ADAPTER_REJECT),
    ]
    assert PRIVATE_SENTINEL not in str(runtime.submissions)


def test_runtime_admission_failures_are_bounded_503(tmp_path: Path) -> None:
    for error_type in (AgentRuntimeQueueFull, AgentRuntimeStopped):
        runtime = AdmissionRuntime(error_type)
        with _client(tmp_path / error_type.__name__, runtime=runtime) as client:
            response = client.post(
                "/api/chat",
                json={"message": PRIVATE_SENTINEL, "attachments": []},
            )
            assert response.status_code == 503
            assert PRIVATE_SENTINEL not in response.text
            assert "test-event" not in response.text


def test_read_only_routes_do_not_submit_events(tmp_path: Path) -> None:
    runtime = RecordingRuntime()
    with _client(tmp_path, runtime=runtime) as client:
        assert client.get("/api/adapters", headers=admin_headers()).status_code == 200
        assert (
            client.get(
                "/api/memory/search", headers=admin_headers(), params={"query": "none"}
            ).status_code
            == 200
        )
    assert runtime.submissions == []


def test_snapshot_restore_precedes_runtime_acceptance(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    order: list[str] = []

    class TrackingStore(AgentStateStore):
        def load(self) -> AgentStateSnapshot:
            order.append("load")
            return super().load()

        def restore_into(self, main_loop, snapshot: AgentStateSnapshot) -> None:
            order.append("restore")
            super().restore_into(main_loop, snapshot)

        def ensure_published(self, snapshot: AgentStateSnapshot) -> None:
            order.append("ensure")
            super().ensure_published(snapshot)

    class TrackingJournal(EventJournal):
        def verify_and_reconcile(self, snapshot_sequence: int, snapshot_hash: str):
            order.append("journal")
            return super().verify_and_reconcile(snapshot_sequence, snapshot_hash)

        def append_v3_migration_checkpoint(self) -> None:
            order.append("v3")
            super().append_v3_migration_checkpoint()

    class TrackingRuntime(RecordingRuntime):
        def start(self) -> None:
            order.append("start")
            super().start()

    store = TrackingStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
    )
    store.save(
        AgentStateSnapshotV2(
            saved_at=datetime.now(timezone.utc),
            last_processed_event_sequence=7,
            emotion_state=EmotionStateSnapshot(
                valence=0.4,
                arousal=0.5,
                optimal_loss=0.6,
            ),
            working_memory=WorkingMemorySnapshot(revision=0, items=()),
        )
    )
    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.agent_state_store = store
    tracking_journal = TrackingJournal(
        settings.event_journal.path,
        settings.event_journal.max_bytes,
        settings.event_journal.retained_files,
    )
    tracking_journal.verify_and_reconcile(7, store.snapshot_hash(store.load()))
    order.clear()
    app.state.event_journal = tracking_journal
    app.state.agent_runtime = TrackingRuntime()

    with TestClient(app) as client:
        assert order == [
            "load",
            "journal",
            "ensure",
            "load",
            "v3",
            "load",
            "restore",
            "start",
        ]
        assert client.app.state.main_loop.emotion_engine.state == EmotionState(
            valence=0.4,
            arousal=0.5,
            optimal_loss=0.6,
        )
        checkpoint = client.app.state.event_journal.records[0]
        assert checkpoint.lifecycle is EventLifecycle.CHECKPOINT
        assert checkpoint.processing_sequence == 7
        assert checkpoint.snapshot_sequence == 7
        assert checkpoint.snapshot_hash == store.snapshot_hash(store.load())
        inspection = client.app.state.event_journal.inspect()
        assert inspection.schema_version == 3
        assert inspection.processing_high_water == 7


def test_v3_migration_failure_prevents_runtime_start(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class MigrationFailingJournal(EventJournal):
        def append_v3_migration_checkpoint(self) -> None:
            raise EventJournalAppendError(
                EventJournalAppendStage.FILE_FSYNC, published=False
            )

    runtime = RecordingRuntime()
    app = create_app(settings)
    app.state.event_journal = MigrationFailingJournal(
        settings.event_journal.path,
        settings.event_journal.max_bytes,
        settings.event_journal.retained_files,
    )
    app.state.agent_runtime = runtime

    with pytest.raises(EventJournalAppendError):
        with TestClient(app):
            pass

    assert runtime.status is AgentRuntimeStatus.CREATED


def test_restored_sequence_continues_and_success_checkpoints_chat(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    store = AgentStateStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
    )
    store.save(
        AgentStateSnapshotV2(
            saved_at=datetime.now(timezone.utc),
            last_processed_event_sequence=7,
            emotion_state=EmotionStateSnapshot(
                valence=0.1,
                arousal=0.2,
                optimal_loss=0.9,
            ),
            working_memory=WorkingMemorySnapshot(revision=0, items=()),
        )
    )

    with _client(tmp_path, settings=settings) as client:
        assert client.app.state.main_loop.emotion_engine.state == EmotionState(
            valence=0.1,
            arousal=0.2,
            optimal_loss=0.9,
        )
        response = client.post(
            "/api/chat",
            json={"message": PRIVATE_SENTINEL, "attachments": [], "debug": False},
        )
        assert response.status_code == 200
        checkpoint = client.app.state.agent_state_store.load()
        assert checkpoint.last_processed_event_sequence == 8
        assert checkpoint.emotion_state == EmotionStateSnapshot(
            valence=client.app.state.main_loop.emotion_engine.state.valence,
            arousal=client.app.state.main_loop.emotion_engine.state.arousal,
            optimal_loss=client.app.state.main_loop.emotion_engine.state.optimal_loss,
        )

    serialized = settings.agent_state.path.read_text(encoding="utf-8")
    assert PRIVATE_SENTINEL not in serialized
    assert "Visible API answer" not in serialized
    assert "prompt" not in serialized.casefold()
    assert "hidden" not in serialized.casefold()
    assert "turns" not in serialized.casefold()


@pytest.mark.parametrize(
    "raw",
    [
        b"{corrupt",
        b'{"schema_version":999}',
        b'{"schema_version":1,"hiddenThought":"PRIVATE-SENTINEL-R02"}',
    ],
)
def test_invalid_snapshot_prevents_runtime_start_and_is_not_overwritten(
    tmp_path: Path, raw: bytes
) -> None:
    settings = _settings(tmp_path)
    settings.agent_state.path.parent.mkdir(parents=True, exist_ok=True)
    settings.agent_state.path.write_bytes(raw)
    runtime = RecordingRuntime()

    with pytest.raises(AgentStateLoadError):
        with _client(tmp_path, settings=settings, runtime=runtime):
            pass

    assert runtime.status is AgentRuntimeStatus.CREATED
    assert settings.agent_state.path.read_bytes() == raw


def test_existing_journal_with_missing_snapshot_is_reconstructed_from_wal(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        response = client.post(
            "/api/chat",
            json={"message": "establish history", "attachments": [], "debug": False},
        )
        assert response.status_code == 200

    expected = client.app.state.agent_state_store.load()
    settings.agent_state.path.unlink()
    app = create_app(settings)

    with TestClient(app) as restarted:
        assert restarted.app.state.agent_state_store.load() == expected
        assert restarted.app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
        assert restarted.app.state.external_reconciliation_required is False
        assert restarted.get("/health").json() == {
            "status": "ok",
            "project": settings.project.name,
        }

    assert settings.agent_state.path.exists()


def test_successful_chat_is_reconstructable_without_private_payloads(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    with _client(tmp_path, settings=settings) as client:
        response = client.post(
            "/api/chat",
            json={"message": PRIVATE_SENTINEL, "attachments": [], "debug": False},
        )
        assert response.status_code == 200
        committed = client.app.state.agent_state_store.load()
        assert (
            client.app.state.state_wal.reconstruct(
                sequence=committed.last_processed_event_sequence
            )
            == committed
        )

    persisted = b"".join(
        path.read_bytes()
        for path in settings.state_wal.directory.rglob("*")
        if path.is_file()
    )
    assert PRIVATE_SENTINEL.encode() not in persisted
    assert b"Visible API answer" not in persisted
    assert b"prompt" not in persisted.lower()
    assert b"hidden" not in persisted.lower()
    assert PRIVATE_SENTINEL not in settings.event_journal.path.read_text()
    assert "Visible API answer" not in settings.event_journal.path.read_text()
    assert PRIVATE_SENTINEL not in settings.agent_state.path.read_text()
    assert "Visible API answer" not in settings.agent_state.path.read_text()


def test_true_rollback_reconciles_external_state_before_runtime_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        client.app.state.memory_system.save_semantic("rollback WM seed")
        assert (
            client.post(
                "/api/chat", json={"message": "rollback WM seed", "attachments": []}
            ).status_code
            == 200
        )

    with _client(tmp_path, settings=settings) as client:
        assert client.app.state.main_loop.working_memory.items
        assert (
            client.post(
                "/api/chat", json={"message": "advance", "attachments": []}
            ).status_code
            == 200
        )
        wal: StateWAL = client.app.state.state_wal
        manifest = wal.inspect().active_manifest
        assert manifest is not None
        generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"

    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[-1])
    transition["record_hash"] = "0" * 64
    lines[-1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    generation.chmod(0o600)
    settings.agent_state.path.unlink()

    def no_startup_replay(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("true-rollback startup replayed Working Memory computation")

    monkeypatch.setattr(DualMemorySystem, "retrieve_context", no_startup_replay)
    monkeypatch.setattr(WorkingMemory, "select", no_startup_replay)
    monkeypatch.setattr(MemoryWorkingMemoryResolver, "resolve", no_startup_replay)
    monkeypatch.setattr(PromptBuilder, "build", no_startup_replay)
    monkeypatch.setattr(ThinkingProvider, "generate", no_startup_replay)

    runtime = RecordingRuntime()
    with _client(tmp_path, settings=settings, runtime=runtime) as reconciled:
        assert runtime.status is AgentRuntimeStatus.ACCEPTING
        assert reconciled.app.state.external_reconciliation_required is False
        inspection = reconciled.app.state.event_journal.inspect()
        assert inspection.schema_version == 3
        assert inspection.terminal_gate_clear is not None
        assert reconciled.app.state.state_wal.inspect().active_manifest is not None
        active_manifest = reconciled.app.state.state_wal.inspect().active_manifest
        assert active_manifest is not None
        assert not active_manifest.external_reconciliation_required
        health = reconciled.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "project": settings.project.name,
        }
        monkeypatch.undo()
        response = reconciled.post(
            "/api/chat", json={"message": "accepted", "attachments": []}
        )
        assert response.status_code == 200

    second_runtime = RecordingRuntime()
    with _client(tmp_path, settings=settings, runtime=second_runtime) as restarted:
        assert second_runtime.status is AgentRuntimeStatus.ACCEPTING
        assert restarted.app.state.event_journal.inspect().schema_version == 3
        manifest = restarted.app.state.state_wal.inspect().active_manifest
        assert manifest is not None
        assert not manifest.external_reconciliation_required


def test_actual_unresolved_path_a_starts_degraded_without_r06_mutation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    class FailingStore(AgentStateStore):
        def save(self, snapshot: AgentStateSnapshot) -> None:
            if snapshot.last_processed_event_sequence > 0:
                raise AgentStateSaveError(
                    AgentStateSaveStage.TEMP_WRITE, published=False
                )
            super().save(snapshot)

    first = create_app(settings)
    first.state.model_provider = ThinkingProvider()
    first.state.memory_system = DualMemorySystem(settings)
    first.state.adapter_registry = AdapterRegistry(settings)
    first.state.agent_state_store = FailingStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
    )
    with TestClient(first) as client:
        assert client.post(
            "/api/chat", json={"message": "unresolved", "attachments": []}
        ).status_code == 500
        transaction = client.app.state.event_journal.inspect().open_transactions[0]
        pending = (
            settings.memory.persist_directory
            / ".r07-episodic-pending"
            / f"{transaction.transaction_id}.json"
        )
    pending.write_text('{"conflict":"well-formed"}')
    pending.chmod(0o600)
    journal_before = settings.event_journal.path.read_bytes()
    snapshot_before = settings.agent_state.path.read_bytes()
    wal_before = {
        path.relative_to(settings.state_wal.directory): path.read_bytes()
        for path in settings.state_wal.directory.rglob("*")
        if path.is_file()
    }

    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    runtime = RecordingRuntime()
    app.state.agent_runtime = runtime

    with TestClient(app) as degraded:
        assert runtime.status is AgentRuntimeStatus.CREATED
        assert degraded.get("/health").json()["status"] == "degraded"
        assert degraded.app.state.event_journal.inspect().open_transactions
        assert degraded.post(
            "/api/chat", json={"message": "blocked", "attachments": []}
        ).status_code == 503
        assert degraded.app.state.main_loop.session_state.turns == []
        assert degraded.app.state.memory_system.db1.get()["ids"] == []
        assert settings.event_journal.path.read_bytes() == journal_before
        assert settings.agent_state.path.read_bytes() == snapshot_before
        assert {
            path.relative_to(settings.state_wal.directory): path.read_bytes()
            for path in settings.state_wal.directory.rglob("*")
            if path.is_file()
        } == wal_before


def test_two_true_rollbacks_preserve_reconciliation_authority(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as first:
        assert first.post(
            "/api/chat", json={"message": "first", "attachments": []}
        ).status_code == 200
        wal: StateWAL = first.app.state.state_wal
        manifest = wal.inspect().active_manifest
        assert manifest is not None
        generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    settings.agent_state.path.unlink()

    with _client(tmp_path, settings=settings) as second:
        assert second.app.state.external_reconciliation_required is False
        assert second.post(
            "/api/chat", json={"message": "second", "attachments": []}
        ).status_code == 200
        wal = second.app.state.state_wal
        manifest = wal.inspect().active_manifest
        assert manifest is not None
        generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    settings.agent_state.path.unlink()

    with _client(tmp_path, settings=settings) as third:
        assert third.app.state.external_reconciliation_required is False
        inspection = third.app.state.event_journal.inspect()
        assert len(inspection.baselines) == 1
        assert len(inspection.completed_startup_reconciliations) == 2
        assert inspection.terminal_gate_clear is not None
        assert third.get("/health").json()["status"] == "ok"


def test_corrupt_wal_keeps_valid_canonical_current_accepting(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        assert client.post(
            "/api/chat", json={"message": "advance", "attachments": []}
        ).status_code == 200
        expected = client.app.state.agent_state_store.load()
        wal: StateWAL = client.app.state.state_wal
        old_manifest = wal.inspect().active_manifest
        assert old_manifest is not None
        generation = (
            wal.root / "generations" / f"{old_manifest.active_generation_id}.jsonl"
        )

    with generation.open("ab") as output:
        output.write(b"corrupt-tail\n")

    with _client(tmp_path, settings=settings) as recovered:
        assert recovered.app.state.agent_state_store.load() == expected
        assert recovered.app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
        assert not (
            recovered.app.state.state_wal.inspect()
            .active_manifest.external_reconciliation_required
        )
        assert (
            recovered.app.state.state_wal.inspect().active_manifest.active_generation_id
            != old_manifest.active_generation_id
        )
        assert recovered.post(
            "/api/chat", json={"message": "continue", "attachments": []}
        ).status_code == 200
        continued = recovered.app.state.agent_state_store.load()
        wal_inspection = recovered.app.state.state_wal.inspect()
        assert continued.last_processed_event_sequence == 2
        assert wal_inspection.latest_snapshot_sequence == 2
        assert wal_inspection.latest_snapshot_hash == (
            recovered.app.state.agent_state_store.snapshot_hash(continued)
        )
    assert generation.read_bytes().endswith(b"corrupt-tail\n")


def test_boot_anchor_failure_prevents_lifespan_readiness(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    atomic_writes = 0

    def fail_anchor(stage: str) -> None:
        nonlocal atomic_writes
        if stage == "temp_write":
            atomic_writes += 1
            if atomic_writes == 2:
                raise OSError("private anchor failure")

    runtime = RecordingRuntime()
    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.agent_runtime = runtime
    wal = StateWAL(settings.state_wal.directory, failure_hook=fail_anchor)
    app.state.state_wal = wal

    with pytest.raises(StateWALError):
        with TestClient(app):
            pass

    assert runtime.status is AgentRuntimeStatus.STOPPED
    assert wal.inspect_boot_anchor_optional() is None


def test_journal_continuity_is_checked_before_v0_snapshot_rewrite(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    legacy = (
        b'{"schema_version":0,"last_event_sequence":3,"emotion":'
        b'{"valence":0.1,"arousal":0.2,"optimal_loss":0.9}}'
    )
    settings.agent_state.path.write_bytes(legacy)
    store = AgentStateStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    journal = EventJournal(
        settings.event_journal.path,
        settings.event_journal.max_bytes,
        settings.event_journal.retained_files,
    )
    journal.verify_and_reconcile(3, "f" * 64)
    journal.close()
    journal_bytes = settings.event_journal.path.read_bytes()
    app = create_app(settings)
    app.state.agent_state_store = store

    with pytest.raises(EventJournalLoadError):
        with TestClient(app):
            pass

    assert not hasattr(app.state, "external_reconciliation_required")
    assert settings.agent_state.path.read_bytes() == legacy
    assert settings.event_journal.path.read_bytes() == journal_bytes


def test_matching_v0_snapshot_is_rewritten_after_journal_reconciliation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    legacy = (
        b'{"schema_version":0,"last_event_sequence":3,"emotion":'
        b'{"valence":0.1,"arousal":0.2,"optimal_loss":0.9}}'
    )
    settings.agent_state.path.write_bytes(legacy)
    store = AgentStateStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    migrated = store.load()
    journal = EventJournal(
        settings.event_journal.path,
        settings.event_journal.max_bytes,
        settings.event_journal.retained_files,
    )
    journal.verify_and_reconcile(3, store.snapshot_hash(migrated))
    journal.close()
    app = create_app(settings)
    app.state.agent_state_store = store

    with TestClient(app) as client:
        assert client.app.state.agent_state_store.load() == migrated

    assert settings.agent_state.path.read_bytes() != legacy
    assert (
        json.loads(settings.agent_state.path.read_text(encoding="utf-8"))[
            "schema_version"
        ]
        == 4
    )


def test_pre_r05_owner_owned_directory_is_hardened_before_startup(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    store = AgentStateStore(
        settings.agent_state.path, settings.emotion.baseline_surprisal
    )
    snapshot = AgentStateSnapshotV2(
        saved_at=datetime.now(timezone.utc),
        last_processed_event_sequence=4,
        emotion_state=EmotionStateSnapshot(
            valence=0.2,
            arousal=0.3,
            optimal_loss=0.8,
        ),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
    )
    store.save(snapshot)
    snapshot_bytes = settings.agent_state.path.read_bytes()
    tmp_path.chmod(0o755)

    with TestClient(create_app(settings)) as client:
        assert client.app.state.agent_state_store.load() == snapshot
        assert tmp_path.stat().st_mode & 0o777 == 0o700
        assert settings.event_journal.path.stat().st_mode & 0o777 == 0o600
        lock_path = tmp_path / f".{settings.event_journal.path.name}.lock"
        assert lock_path.stat().st_mode & 0o777 == 0o600

    assert settings.agent_state.path.read_bytes() == snapshot_bytes


def test_snapshot_does_not_shadow_memory_or_adapter_registry(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        memory_marker = "MEMORY-AUTHORITY-MARKER"
        adapter_marker = "ADAPTER-AUTHORITY-MARKER"
        client.app.state.memory_system.save_episodic(memory_marker, "visible")
        client.app.state.adapter_registry.register_candidate(
            adapter_id=adapter_marker,
            adapter_path=tmp_path / "adapter",
            dataset_path=tmp_path / "dataset.jsonl",
            dataset_hash="dataset-hash-marker",
        )
        response = client.post(
            "/api/chat",
            json={"message": "checkpoint", "attachments": [], "debug": False},
        )
        assert response.status_code == 200

    serialized = settings.agent_state.path.read_text(encoding="utf-8")
    assert memory_marker not in serialized
    assert adapter_marker not in serialized
    assert "dataset-hash-marker" not in serialized
    assert settings.adapter_registry.path.exists()
    assert settings.memory.persist_directory.exists()


@pytest.mark.parametrize(
    "save_stage",
    [AgentStateSaveStage.TEMP_WRITE, AgentStateSaveStage.ATOMIC_REPLACE],
)
def test_snapshot_checkpoint_failure_returns_bounded_indeterminate_500(
    tmp_path: Path, save_stage: AgentStateSaveStage
) -> None:
    settings = _settings(tmp_path)

    class FailingStore(AgentStateStore):
        def save(self, snapshot: AgentStateSnapshot) -> None:
            if snapshot.last_processed_event_sequence > 1:
                raise AgentStateSaveError(
                    save_stage,
                    published=False,
                )
            super().save(snapshot)

    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.agent_state_store = FailingStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
    )

    with TestClient(app) as client:
        client.app.state.memory_system.save_semantic("U5 durable WM seed")
        assert client.post(
            "/api/chat", json={"message": "U5 durable WM seed", "attachments": []}
        ).status_code == 200
        prior = client.app.state.agent_state_store.load()
        prior_wm = (
            client.app.state.main_loop.working_memory.revision,
            client.app.state.main_loop.working_memory.items,
        )
        prior_bytes = settings.agent_state.path.read_bytes()
        prior_hash = client.app.state.agent_state_store.snapshot_hash(prior)
        response = client.post(
            "/api/chat",
            json={"message": "visible request", "attachments": [], "debug": False},
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Agent mutation durability is indeterminate"
        }
        assert PRIVATE_SENTINEL not in response.text
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert (
            client.app.state.agent_state_store.load().last_processed_event_sequence == 1
        )
        assert client.app.state.state_wal.inspect().latest_snapshot_sequence == 2
        assert len(client.app.state.main_loop.session_state.turns) == 1
        assert len(client.app.state.memory_system.db1.get()["ids"]) == 1
        candidate_wm = (
            client.app.state.main_loop.working_memory.revision,
            client.app.state.main_loop.working_memory.items,
        )
        assert candidate_wm != prior_wm
        wal_candidate = client.app.state.state_wal.reconstruct(sequence=2)
        assert wal_candidate.working_memory.revision == candidate_wm[0]
        assert tuple(
            (item.item_id, item.source_kind, item.source_id)
            for item in wal_candidate.working_memory.items
        ) == tuple(
            (item.item_id, item.source_kind.value, item.source_id)
            for item in candidate_wm[1]
        )
        assert client.app.state.agent_state_store.load() == prior
        assert settings.agent_state.path.read_bytes() == prior_bytes
        assert (
            client.app.state.state_wal.inspect().latest_snapshot_hash != prior_hash
        )
        transaction = client.app.state.event_journal.inspect().open_transactions[0]
        assert transaction.terminal_lifecycle is None
        pending_path = (
            settings.memory.persist_directory
            / ".r07-episodic-pending"
            / f"{transaction.transaction_id}.json"
        )
        assert pending_path.exists()
        assert PRIVATE_SENTINEL not in pending_path.read_text()
        assert PRIVATE_SENTINEL not in settings.event_journal.path.read_text()
        assert PRIVATE_SENTINEL not in settings.agent_state.path.read_text()
        assert all(
            PRIVATE_SENTINEL.encode() not in path.read_bytes()
            for path in settings.state_wal.directory.iterdir()
            if path.is_file()
        )

    with _client(tmp_path, settings=settings) as recovered:
        inspection = recovered.app.state.event_journal.inspect()
        assert not inspection.open_transactions
        assert inspection.aborted_transactions
        assert not pending_path.exists()
        assert recovered.app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
        assert recovered.app.state.state_wal.inspect().latest_snapshot_sequence == 1
        assert recovered.app.state.main_loop.session_state.turns == []
        assert len(recovered.app.state.memory_system.db1.get()["ids"]) == 1
        assert (
            recovered.app.state.main_loop.working_memory.revision,
            recovered.app.state.main_loop.working_memory.items,
        ) == prior_wm
        assert recovered.app.state.agent_state_store.snapshot_hash(
            recovered.app.state.agent_state_store.load()
        ) == prior_hash


def test_wal_failure_after_prepared_prevents_snapshot_publish_and_fail_stops(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    def fail_transition(stage: str) -> None:
        if stage == "transition_fsync":
            raise OSError("private failure detail")

    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.state_wal = StateWAL(
        settings.state_wal.directory, failure_hook=fail_transition
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/chat", json={"message": PRIVATE_SENTINEL, "attachments": []}
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Agent mutation durability is indeterminate"
        }
        assert "private failure detail" not in response.text
        assert PRIVATE_SENTINEL not in response.text
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert (
            client.app.state.agent_state_store.load().last_processed_event_sequence == 0
        )
        assert (
            client.app.state.event_journal.records[-1].lifecycle
            is EventLifecycle.PREPARED
        )


def test_handler_failure_restores_r04_state_records_failed_and_continues(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    initial_emotion = EmotionState(valence=0.0, arousal=0.0, optimal_loss=1.0)
    app = create_app(settings)
    app.state.model_provider = FailOnceAfterEmotionProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)

    with TestClient(app) as client:
        main_loop = client.app.state.main_loop
        before_temporal = main_loop.emotion_engine.temporal_state
        before_calibration = main_loop.loss_calibration.export()
        before_working_memory = (
            main_loop.working_memory.revision,
            main_loop.working_memory.items,
        )
        with pytest.raises(ValueError, match=PRIVATE_SENTINEL):
            client.post(
                "/api/chat",
                json={"message": PRIVATE_SENTINEL, "attachments": [], "debug": False},
            )

        assert main_loop.emotion_engine.state == initial_emotion
        assert main_loop.emotion_engine.temporal_state == before_temporal
        assert main_loop.loss_calibration.export() == before_calibration
        assert (
            main_loop.working_memory.revision,
            main_loop.working_memory.items,
        ) == before_working_memory
        assert main_loop.context_registry.state.frames == ()
        assert main_loop.context_registry.current_context_id is None
        assert main_loop.session_state.turns == []
        assert client.app.state.memory_system.db1.get(
            include=["documents", "metadatas"]
        )["ids"] == []
        failed = client.app.state.event_journal.records[-1]
        assert failed.lifecycle is EventLifecycle.FAILED
        assert failed.processing_sequence == 1
        assert failed.snapshot_sequence == 0
        assert failed.failure_category is EventFailureCategory.HANDLER_FAILURE

        response = client.post(
            "/api/chat",
            json={"message": "retry", "attachments": [], "debug": False},
        )
        assert response.status_code == 200
        completed = client.app.state.event_journal.records[-1]
        assert completed.lifecycle is EventLifecycle.COMPLETED
        assert completed.processing_sequence == 2
        assert (
            client.app.state.agent_state_store.load().last_processed_event_sequence == 2
        )

    journal_bytes = settings.event_journal.path.read_text(encoding="utf-8")
    assert PRIVATE_SENTINEL not in journal_bytes
    assert "prompt" not in journal_bytes.casefold()
    assert "message" not in journal_bytes.casefold()


def test_handler_failure_rolls_back_elapsed_structured_state_and_success_artifacts(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    provider = FailOnSecondGenerationProvider()

    with _client(tmp_path, settings=settings, provider=provider) as client:
        main_loop = client.app.state.main_loop
        clock = {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)}
        setattr(main_loop.emotion_engine, "_clock", lambda: clock["now"])

        first = client.post(
            "/api/chat", json={"message": "baseline", "attachments": []}
        )
        assert first.status_code == 200
        baseline_snapshot = client.app.state.agent_state_store.load()
        before_emotion = main_loop.emotion_engine.state
        before_temporal = main_loop.emotion_engine.temporal_state
        before_calibration = main_loop.loss_calibration.export()
        before_working_memory = (
            main_loop.working_memory.revision,
            main_loop.working_memory.items,
        )
        before_context = main_loop.context_registry.state
        before_turns = tuple(main_loop.session_state.turns)
        before_episode_ids = set(
            client.app.state.memory_system.db1.get()["ids"]
        )

        clock["now"] += timedelta(seconds=60)
        with pytest.raises(ValueError, match=PRIVATE_SENTINEL):
            client.post(
                "/api/chat", json={"message": "failed", "attachments": []}
            )

        assert main_loop.emotion_engine.state == before_emotion
        assert main_loop.emotion_engine.temporal_state == before_temporal
        assert main_loop.loss_calibration.export() == before_calibration
        assert (
            main_loop.working_memory.revision,
            main_loop.working_memory.items,
        ) == before_working_memory
        assert main_loop.context_registry.state == before_context
        assert tuple(main_loop.session_state.turns) == before_turns
        assert set(client.app.state.memory_system.db1.get()["ids"]) == (
            before_episode_ids
        )
        assert client.app.state.agent_state_store.load() == baseline_snapshot
        failed = client.app.state.event_journal.records[-1]
        assert failed.lifecycle is EventLifecycle.FAILED
        assert failed.processing_sequence == 2
        assert failed.failure_category is EventFailureCategory.HANDLER_FAILURE

        retry = client.post(
            "/api/chat", json={"message": "retry", "attachments": []}
        )
        assert retry.status_code == 200


def test_handler_context_mutation_failure_restores_current_and_has_no_success(
    tmp_path: Path,
) -> None:
    """A Context mutation made before model failure is inside the rollback boundary."""
    settings = _settings(tmp_path)
    app = create_app(settings)
    app.state.model_provider = FailOnceAfterEmotionProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)

    with TestClient(app) as client:
        with pytest.raises(ValueError, match=PRIVATE_SENTINEL):
            client.post(
                "/api/chat",
                json={
                    "message": PRIVATE_SENTINEL,
                    "attachments": [],
                    "client_session_id": "failed-context-session",
                    "debug": False,
                },
            )

        registry = client.app.state.main_loop.context_registry
        assert registry.state.frames == ()
        assert registry.current_context_id is None
        records = client.app.state.event_journal.inspect().records
        assert records[-1].lifecycle is EventLifecycle.FAILED
        assert records[-1].failure_category is EventFailureCategory.HANDLER_FAILURE
        assert not any(record.lifecycle is EventLifecycle.COMPLETED for record in records)
        assert PRIVATE_SENTINEL not in client.app.state.event_journal.path.read_text()


def test_handler_failure_restore_failure_enters_fail_stop(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class RestoreFailingStore(AgentStateStore):
        restore_calls = 0

        def restore_into(self, main_loop, snapshot: AgentStateSnapshot) -> None:
            self.restore_calls += 1
            if self.restore_calls > 1:
                raise AgentStateLoadError("bounded restore failure")
            super().restore_into(main_loop, snapshot)

    app = create_app(settings)
    app.state.model_provider = FailOnceAfterEmotionProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.agent_state_store = RestoreFailingStore(
        settings.agent_state.path, settings.emotion.baseline_surprisal
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "fail", "attachments": [], "debug": False},
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Agent mutation durability is indeterminate"
        }
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert (
            client.post(
                "/api/chat",
                json={"message": "later", "attachments": [], "debug": False},
            ).status_code
            == 503
        )


def test_failed_append_failure_enters_fail_stop(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class FailedAppendJournal(EventJournal):
        def append_failed(
            self,
            event: AgentEvent,
            snapshot_sequence: int,
            snapshot_hash: str,
        ) -> None:
            raise EventJournalAppendError(
                EventJournalAppendStage.FILE_FSYNC, published=False
            )

    app = create_app(settings)
    app.state.model_provider = FailOnceAfterEmotionProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.event_journal = FailedAppendJournal(
        settings.event_journal.path,
        settings.event_journal.max_bytes,
        settings.event_journal.retained_files,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "fail", "attachments": [], "debug": False},
        )
        assert response.status_code == 500
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED


def test_accepted_append_failure_returns_bounded_503_without_handler(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    class AcceptedAppendJournal(EventJournal):
        def append_accepted(self, event: AgentEvent) -> None:
            raise EventJournalAppendError(
                EventJournalAppendStage.FILE_FSYNC, published=False
            )

    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.event_journal = AcceptedAppendJournal(
        settings.event_journal.path,
        settings.event_journal.max_bytes,
        settings.event_journal.retained_files,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": PRIVATE_SENTINEL, "attachments": [], "debug": False},
        )
        assert response.status_code == 503
        assert response.json() == {
            "detail": "Agent runtime durability is temporarily unavailable"
        }
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert [
            record.lifecycle for record in client.app.state.event_journal.records
        ] == [
            EventLifecycle.CHECKPOINT,
            EventLifecycle.CHECKPOINT,
            EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED,
        ]
        assert client.app.state.main_loop.session_state.turns == []


def test_memory_prepare_failure_aborts_before_internal_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    original_prepare = MemoryEpisodicParticipant.prepare

    def fail_prepare(
        participant: MemoryEpisodicParticipant, binding: TransactionBinding
    ) -> NoReturn:
        original_prepare(participant, binding)
        raise OSError("private Memory prepare failure")

    monkeypatch.setattr(MemoryEpisodicParticipant, "prepare", fail_prepare)
    with _client(tmp_path, settings=settings) as client:
        response = client.post(
            "/api/chat", json={"message": "prepare", "attachments": []}
        )

        assert response.status_code == 500
        assert response.json() == {
            "detail": "Agent mutation durability is indeterminate"
        }
        assert client.app.state.agent_state_store.load().last_processed_event_sequence == 0
        assert client.app.state.memory_system.db1.get()["ids"] == []
        assert client.app.state.main_loop.session_state.turns == []
        inspection = client.app.state.event_journal.inspect()
        assert inspection.aborted_transactions[0].abort_outcomes == (
            ("memory.episodic", AbortOutcome.ABORTED),
        )
        assert not any(
            record.lifecycle is EventLifecycle.PREPARED
            for record in inspection.records
        )


def test_session_finalize_failure_preserves_finalized_memory_and_internal_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)

    def fail_session(
        _participant: SessionTurnParticipant, _binding: object
    ) -> NoReturn:
        raise OSError("private Session finalize failure")

    monkeypatch.setattr(SessionTurnParticipant, "finalize", fail_session)
    with _client(tmp_path, settings=settings) as client:
        response = client.post(
            "/api/chat", json={"message": "partial", "attachments": []}
        )

        assert response.status_code == 500
        assert response.json() == {
            "detail": "Agent mutation durability is indeterminate"
        }
        assert client.app.state.agent_state_store.load().last_processed_event_sequence == 1
        assert client.app.state.state_wal.inspect().latest_snapshot_sequence == 1
        candidate = client.app.state.agent_state_store.load()
        candidate_wm = (
            client.app.state.main_loop.working_memory.revision,
            client.app.state.main_loop.working_memory.items,
        )
        candidate_hash = client.app.state.agent_state_store.snapshot_hash(candidate)
        episode_count = len(client.app.state.memory_system.db1.get()["ids"])
        assert client.app.state.main_loop.session_state.turns == []
        assert len(client.app.state.memory_system.db1.get()["ids"]) == 1
        transaction = client.app.state.event_journal.inspect().open_transactions[0]
        assert transaction.participant_outcomes == (
            ("memory.episodic", ParticipantOutcome.FINALIZED),
        )
        assert transaction.unresolved_participants == ("session.turn",)
        assert transaction.reconciliation_reason is not None
        assert not any(
            record.lifecycle is EventLifecycle.COMPLETED
            for record in client.app.state.event_journal.records
        )

    with _client(tmp_path, settings=settings) as reconciled:
        restored = reconciled.app.state.agent_state_store.load()
        assert restored == candidate
        assert (
            reconciled.app.state.agent_state_store.snapshot_hash(restored)
            == candidate_hash
        )
        assert (
            reconciled.app.state.main_loop.working_memory.revision,
            reconciled.app.state.main_loop.working_memory.items,
        ) == candidate_wm
        assert len(reconciled.app.state.memory_system.db1.get()["ids"]) == episode_count
        assert not reconciled.app.state.event_journal.inspect().open_transactions


def test_finalization_failure_preserves_internal_commit_without_restore(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    class TrackingStore(AgentStateStore):
        restore_calls = 0

        def restore_into(self, main_loop, snapshot: AgentStateSnapshot) -> None:
            self.restore_calls += 1
            super().restore_into(main_loop, snapshot)

    class FinalizationFailingRuntime(RecordingRuntime):
        def configure_durability(self, **kwargs: Any) -> None:
            def fail_finalization(
                _event: AgentEvent, _evidence: object
            ) -> NoReturn:
                raise OSError("private finalization failure")

            kwargs["finalization_checkpoint"] = fail_finalization
            super().configure_durability(**kwargs)

    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.agent_state_store = TrackingStore(
        settings.agent_state.path, settings.emotion.baseline_surprisal
    )
    app.state.agent_runtime = FinalizationFailingRuntime()

    with TestClient(app) as client:
        response = client.post(
            "/api/chat", json={"message": PRIVATE_SENTINEL, "attachments": []}
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Agent mutation durability is indeterminate"
        }
        assert PRIVATE_SENTINEL not in response.text
        assert "private finalization failure" not in response.text
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert client.app.state.agent_state_store.load().last_processed_event_sequence == 1
        assert client.app.state.agent_state_store.restore_calls == 1
        assert client.app.state.state_wal.inspect().latest_snapshot_sequence == 1
        records = client.app.state.event_journal.records
        assert records[-1].lifecycle is EventLifecycle.PREPARED
        assert not any(
            record.lifecycle is EventLifecycle.COMPLETED for record in records
        )


def test_completed_append_failure_is_reconciled_as_committed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class CompletedAppendJournal(EventJournal):
        def append_completed(
            self,
            event: AgentEvent,
            snapshot_sequence: int,
            snapshot_hash: str,
            **_kwargs: object,
        ) -> None:
            raise EventJournalAppendError(
                EventJournalAppendStage.FILE_FSYNC, published=False
            )

    app = create_app(settings)
    app.state.model_provider = ThinkingProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    app.state.event_journal = CompletedAppendJournal(
        settings.event_journal.path,
        settings.event_journal.max_bytes,
        settings.event_journal.retained_files,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "commit", "attachments": [], "debug": False},
        )
        assert response.status_code == 500
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        candidate = client.app.state.agent_state_store.load()
        candidate_wm = (
            client.app.state.main_loop.working_memory.revision,
            client.app.state.main_loop.working_memory.items,
        )
        candidate_hash = client.app.state.agent_state_store.snapshot_hash(candidate)
        assert (
            candidate.last_processed_event_sequence == 1
        )
        assert client.app.state.state_wal.inspect().latest_snapshot_sequence == 1
        assert client.app.state.event_journal.records[-1].lifecycle is (
            EventLifecycle.TRANSACTION_COMPLETED
        )

    with _client(tmp_path, settings=settings) as restarted:
        assert (
            restarted.app.state.agent_state_store.load().last_processed_event_sequence
            == 1
        )
        restored = restarted.app.state.agent_state_store.load()
        assert (
            restarted.app.state.main_loop.working_memory.revision,
            restarted.app.state.main_loop.working_memory.items,
        ) == candidate_wm
        assert (
            restarted.app.state.agent_state_store.snapshot_hash(restored)
            == candidate_hash
        )
        assert restarted.app.state.event_journal.inspect().processing_high_water == 1
        assert restarted.app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
        records = restarted.app.state.event_journal.records
        assert records[-2].lifecycle is EventLifecycle.RECOVERY_CLASSIFIED
        assert (
            records[-2].failure_category
            is EventFailureCategory.COMMITTED_BEFORE_CRASH
        )
        assert records[-1].lifecycle is EventLifecycle.CHECKPOINT
        assert records[-1].wal_record_id is not None


def test_second_startup_cannot_touch_snapshot_before_journal_lease(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    first_app = create_app(settings)
    first_app.state.model_provider = ThinkingProvider()
    first_app.state.memory_system = DualMemorySystem(settings)
    first_app.state.adapter_registry = AdapterRegistry(settings)

    with TestClient(first_app):
        original = settings.agent_state.path.read_bytes()
        load_called = False

        class TrackingStore(AgentStateStore):
            def load(self) -> AgentStateSnapshot:
                nonlocal load_called
                load_called = True
                return super().load()

        second_app = create_app(settings)
        second_app.state.model_provider = ThinkingProvider()
        second_app.state.memory_system = DualMemorySystem(settings)
        second_app.state.adapter_registry = AdapterRegistry(settings)
        second_app.state.agent_state_store = TrackingStore(
            settings.agent_state.path, settings.emotion.baseline_surprisal
        )

        with pytest.raises(EventJournalLoadError):
            with TestClient(second_app):
                pass

        assert not load_called
        assert settings.agent_state.path.read_bytes() == original


def test_chat_commits_post_chat_working_memory_in_agent_state_v4(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    resolved_body = "U4-RESOLVED-BODY-SENTINEL"

    with _client(tmp_path, settings=settings) as client:
        semantic_id = client.app.state.memory_system.save_semantic(resolved_body)
        response = client.post(
            "/api/chat",
            json={"message": resolved_body, "attachments": [], "debug": False},
        )

        assert response.status_code == 200
        assert set(response.json()) == {"episode_id", "response", "emotion", "model"}
        snapshot = client.app.state.agent_state_store.load()
        authoritative_items = client.app.state.main_loop.working_memory.items
        assert snapshot.schema_version == 4
        assert snapshot.working_memory.revision == (
            client.app.state.main_loop.working_memory.revision
        )
        assert len(snapshot.working_memory.items) == len(authoritative_items) == 1
        assert snapshot.working_memory.items[0].source_kind == (
            WorkingMemorySourceKind.SEMANTIC.value
        )
        assert snapshot.working_memory.items[0].source_id == semantic_id
        assert snapshot.working_memory.items[0].item_id == authoritative_items[0].item_id
        assert snapshot.working_memory.items[0].activation == authoritative_items[0].activation
        assert snapshot.working_memory.items[0].salience == authoritative_items[0].salience
        assert snapshot.working_memory.items[0].created_revision == (
            authoritative_items[0].created_revision
        )
        assert snapshot.working_memory.items[0].last_activated_revision == (
            authoritative_items[0].last_activated_revision
        )
        assert resolved_body not in response.text
        assert "working_memory" not in response.text
        assert "debug" not in response.text

        durable = (
            settings.agent_state.path.read_bytes()
            + settings.event_journal.path.read_bytes()
            + b"".join(
                path.read_bytes()
                for path in settings.state_wal.directory.rglob("*")
                if path.is_file()
            )
        )
        assert resolved_body.encode() not in durable


def test_debug_projection_can_use_resolved_body_without_durable_leak(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    resolved_body = "U4-DEBUG-RESOLVED-BODY-SENTINEL"

    with _client(tmp_path, settings=settings) as client:
        client.app.state.memory_system.save_semantic(resolved_body)
        response = client.post(
            "/api/chat/debug",
            headers=admin_headers(),
            json={"message": resolved_body, "attachments": [], "debug": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert resolved_body in data["prompt"]
        assert resolved_body in str(data["retrieved_memory"])
        durable = (
            settings.agent_state.path.read_bytes()
            + settings.event_journal.path.read_bytes()
            + b"".join(
                path.read_bytes()
                for path in settings.state_wal.directory.rglob("*")
                if path.is_file()
            )
        )
        assert resolved_body.encode() not in durable
        assert resolved_body.encode() not in settings.agent_state.path.read_bytes()


def test_handler_failure_restores_prior_canonical_working_memory(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        client.app.state.memory_system.save_semantic("U4 failure checkpoint marker")
        first = client.post(
            "/api/chat",
            json={"message": "U4 failure checkpoint marker", "attachments": []},
        )
        assert first.status_code == 200
        prior = client.app.state.agent_state_store.load()
        prior_working_memory = (
            client.app.state.main_loop.working_memory.revision,
            client.app.state.main_loop.working_memory.items,
        )
        prior_canonical = client.app.state.agent_state_store.canonical_bytes(prior)
        prior_hash = client.app.state.agent_state_store.snapshot_hash(prior)

        failing = FailOnceAfterEmotionProvider()
        client.app.state.main_loop.agent.provider = failing
        with pytest.raises(ValueError, match=PRIVATE_SENTINEL):
            client.post(
                "/api/chat",
                json={"message": "U4 failure checkpoint marker", "attachments": []},
            )

        assert (
            client.app.state.main_loop.working_memory.revision,
            client.app.state.main_loop.working_memory.items,
        ) == prior_working_memory
        assert client.app.state.agent_state_store.load() == prior
        assert client.app.state.agent_state_store.canonical_bytes(
            client.app.state.agent_state_store.load()
        ) == prior_canonical
        assert client.app.state.agent_state_store.snapshot_hash(
            client.app.state.agent_state_store.load()
        ) == prior_hash
        failed = client.app.state.event_journal.records[-1]
        assert failed.lifecycle is EventLifecycle.FAILED
        assert failed.failure_category is EventFailureCategory.HANDLER_FAILURE
        assert failed.snapshot_hash == prior_hash
        assert sum(
            record.lifecycle is EventLifecycle.FAILED
            for record in client.app.state.event_journal.records
        ) == 1
        durable = b"".join(
            path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
        assert PRIVATE_SENTINEL.encode() not in durable


def test_finalized_episode_is_retrieved_and_admitted_on_later_turn(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        first = client.post(
            "/api/chat",
            json={"message": "U4 later-turn retrieval marker", "attachments": []},
        )
        assert first.status_code == 200
        episode_id = first.json()["episode_id"]
        assert client.app.state.memory_system.get_episodic_record(episode_id) is not None

        second = client.post(
            "/api/chat/debug",
            headers=admin_headers(),
            json={
                "message": "U4 later-turn retrieval marker",
                "attachments": [],
                "debug": True,
            },
        )
        assert second.status_code == 200
        assert episode_id in str(second.json()["retrieved_memory"])
        assert (
            "User: U4 later-turn retrieval marker\nAssistant: "
            "Visible API answer."
        ) in second.json()["prompt"]
        assert any(
            item.source_kind is WorkingMemorySourceKind.EPISODIC
            and item.source_id == episode_id
            for item in client.app.state.main_loop.working_memory.items
        )


@pytest.mark.parametrize(
    "status",
    [
        WorkingMemoryResolutionStatus.MISSING,
        WorkingMemoryResolutionStatus.ARCHIVED,
        WorkingMemoryResolutionStatus.UNAVAILABLE,
    ],
)
def test_startup_does_not_resolve_noneligible_working_memory_refs(
    tmp_path: Path,
    status: WorkingMemoryResolutionStatus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    marker = f"U5-{status.value}-working-memory-ref"
    with _client(tmp_path, settings=settings) as first:
        source_id = first.app.state.memory_system.save_semantic(marker)
        assert first.post(
            "/api/chat", json={"message": marker, "attachments": []}
        ).status_code == 200
        assert any(
            item.source_id == source_id
            for item in first.app.state.main_loop.working_memory.items
        )

    original_resolve = MemoryWorkingMemoryResolver.resolve
    startup_calls = 0

    def fail_if_startup_resolves(
        resolver: MemoryWorkingMemoryResolver, item: object
    ) -> NoReturn:
        nonlocal startup_calls
        startup_calls += 1
        raise AssertionError("startup resolved a Working Memory reference")

    monkeypatch.setattr(
        MemoryWorkingMemoryResolver, "resolve", fail_if_startup_resolves
    )
    with _client(tmp_path, settings=settings) as restarted:
        assert startup_calls == 0
        resolver = restarted.app.state.main_loop.working_memory_resolver
        resolver.resolve = (  # type: ignore[method-assign]
            lambda _item: WorkingMemoryResolution(status)
        )
        monkeypatch.setattr(
            restarted.app.state.memory_system,
            "retrieve_context",
            lambda _query: MemoryContext(),
        )
        expected_reason = {
            WorkingMemoryResolutionStatus.MISSING: (
                WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE
            ),
            WorkingMemoryResolutionStatus.ARCHIVED: (
                WorkingMemoryDecisionReason.SOURCE_ARCHIVED
            ),
            WorkingMemoryResolutionStatus.UNAVAILABLE: (
                WorkingMemoryDecisionReason.SOURCE_UNAVAILABLE
            ),
        }[status]
        view = restarted.app.state.main_loop.working_memory.select(resolver)
        assert view.decisions[0].reason is expected_reason
        response = restarted.post(
            "/api/chat/debug",
            headers=admin_headers(),
            json={"message": "later decision", "attachments": [], "debug": True},
        )
        assert response.status_code == 200
        assert marker not in response.json()["prompt"]
        assert marker not in str(response.json()["retrieved_memory"])
        assert all(
            item.source_id == source_id
            for item in restarted.app.state.main_loop.working_memory.items
        )
        resolver.resolve = original_resolve.__get__(resolver, type(resolver))


def _client(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    configure_admin_token: bool = True,
    runtime: AgentRuntime | AdmissionRuntime | None = None,
    provider: DummyProvider | None = None,
) -> TestClient:
    if configure_admin_token:
        os.environ["KAGYA_TEST_ADMIN_TOKEN"] = ADMIN_TOKEN
    else:
        os.environ.pop("KAGYA_TEST_ADMIN_TOKEN", None)
    app_settings = settings or _settings(tmp_path)
    app = create_app(app_settings)
    app.state.model_provider = provider or ThinkingProvider()
    app.state.memory_system = DualMemorySystem(app_settings)
    app.state.adapter_registry = AdapterRegistry(app_settings)
    if runtime is not None:
        app.state.agent_runtime = runtime
    return TestClient(app)


def _settings(tmp_path: Path) -> Settings:
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp_path.chmod(0o700)
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
            "agent_state": settings.agent_state.model_copy(
                update={"path": tmp_path / "agent_state.json"}
            ),
            "event_journal": settings.event_journal.model_copy(
                update={"path": tmp_path / "event_journal.jsonl"}
            ),
            "state_wal": settings.state_wal.model_copy(
                update={"directory": tmp_path / "private" / "state_wal"}
            ),
        }
    )


def admin_headers() -> dict[str, str]:
    return {"X-KAGYA-Admin-Token": ADMIN_TOKEN}
