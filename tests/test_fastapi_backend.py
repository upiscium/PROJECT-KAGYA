import json
import os
from collections.abc import Callable
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from kagya.api.server import create_app
from kagya.body import EmotionState
from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry
from kagya.memory import DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import (
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
    AgentStateSnapshot,
    AgentStateStore,
    EmotionStateSnapshot,
    EventFailureCategory,
    EventJournal,
    EventJournalAppendError,
    EventJournalAppendStage,
    EventJournalLoadError,
    EventLifecycle,
    StateWAL,
    StateWALError,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ADMIN_TOKEN = "test-admin-token"
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-R02"


class ThinkingProvider(DummyProvider):
    response_text = f"<think>{PRIVATE_SENTINEL}</think>Visible API answer."


class FailOnceAfterEmotionProvider(ThinkingProvider):
    def __init__(self) -> None:
        self.failed = False

    def generate(self, prompt: str) -> str:
        if not self.failed:
            self.failed = True
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


def test_api_chat_works_with_dummy_provider_without_debug_leak(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
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
    with _client(tmp_path) as client:
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
        stored = client.app.state.memory_system.db1.get(
            ids=[data["episode_id"]], include=["documents", "metadatas"]
        )
        assert PRIVATE_SENTINEL not in str(stored)
        assert "hidden_thought" not in stored["metadatas"][0]


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

    class TrackingRuntime(RecordingRuntime):
        def start(self) -> None:
            order.append("start")
            super().start()

    store = TrackingStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
    )
    store.save(
        AgentStateSnapshot(
            saved_at=datetime.now(timezone.utc),
            last_processed_event_sequence=7,
            emotion_state=EmotionStateSnapshot(
                valence=0.4,
                arousal=0.5,
                optimal_loss=0.6,
            ),
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
        assert order == ["load", "journal", "ensure", "restore", "start"]
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


def test_restored_sequence_continues_and_success_checkpoints_chat(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    store = AgentStateStore(
        settings.agent_state.path,
        settings.emotion.baseline_surprisal,
    )
    store.save(
        AgentStateSnapshot(
            saved_at=datetime.now(timezone.utc),
            last_processed_event_sequence=7,
            emotion_state=EmotionStateSnapshot(
                valence=0.1,
                arousal=0.2,
                optimal_loss=0.9,
            ),
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


def test_true_rollback_keeps_runtime_reconciliation_gated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
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

    with generation.open("ab") as output:
        output.write(b"corrupt-tail\n")
    settings.agent_state.path.unlink()

    runtime = RecordingRuntime()
    with _client(tmp_path, settings=settings, runtime=runtime) as gated:
        assert runtime.status is AgentRuntimeStatus.CREATED
        assert gated.app.state.state_wal.inspect().active_manifest is not None
        assert gated.app.state.state_wal.inspect().active_manifest.external_reconciliation_required
        response = gated.post(
            "/api/chat", json={"message": "blocked", "attachments": []}
        )
        assert response.status_code == 503

    second_runtime = RecordingRuntime()
    with _client(tmp_path, settings=settings, runtime=second_runtime) as still_gated:
        assert second_runtime.status is AgentRuntimeStatus.CREATED
        manifest = still_gated.app.state.state_wal.inspect().active_manifest
        assert manifest is not None
        assert manifest.external_reconciliation_required
        assert still_gated.post(
            "/api/chat", json={"message": "still-blocked", "attachments": []}
        ).status_code == 503


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
        == 1
    )


def test_pre_r05_owner_owned_directory_is_hardened_before_startup(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    store = AgentStateStore(
        settings.agent_state.path, settings.emotion.baseline_surprisal
    )
    snapshot = AgentStateSnapshot(
        saved_at=datetime.now(timezone.utc),
        last_processed_event_sequence=4,
        emotion_state=EmotionStateSnapshot(
            valence=0.2,
            arousal=0.3,
            optimal_loss=0.8,
        ),
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


def test_snapshot_checkpoint_failure_returns_bounded_indeterminate_500(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    class FailingStore(AgentStateStore):
        def save(self, snapshot: AgentStateSnapshot) -> None:
            if snapshot.last_processed_event_sequence > 0:
                raise AgentStateSaveError(
                    AgentStateSaveStage.TEMP_WRITE,
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
        response = client.post(
            "/api/chat",
            json={"message": PRIVATE_SENTINEL, "attachments": [], "debug": False},
        )
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Agent mutation durability is indeterminate"
        }
        assert PRIVATE_SENTINEL not in response.text
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert (
            client.app.state.agent_state_store.load().last_processed_event_sequence == 0
        )
        assert client.app.state.state_wal.inspect().latest_snapshot_sequence == 1

    with _client(tmp_path, settings=settings) as restarted:
        assert (
            restarted.app.state.agent_state_store.load().last_processed_event_sequence
            == 0
        )
        response = restarted.post(
            "/api/chat", json={"message": "next", "attachments": []}
        )
        assert response.status_code == 200
        assert (
            restarted.app.state.agent_state_store.load().last_processed_event_sequence
            == 2
        )


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
        with pytest.raises(ValueError, match=PRIVATE_SENTINEL):
            client.post(
                "/api/chat",
                json={"message": PRIVATE_SENTINEL, "attachments": [], "debug": False},
            )

        assert client.app.state.main_loop.emotion_engine.state == initial_emotion
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
        ] == [EventLifecycle.CHECKPOINT]
        assert client.app.state.main_loop.session_state.turns == []


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

    with _client(tmp_path, settings=settings) as restarted:
        assert (
            restarted.app.state.agent_state_store.load().last_processed_event_sequence
            == 1
        )
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


def _client(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    configure_admin_token: bool = True,
    runtime: AgentRuntime | AdmissionRuntime | None = None,
) -> TestClient:
    if configure_admin_token:
        os.environ["KAGYA_TEST_ADMIN_TOKEN"] = ADMIN_TOKEN
    else:
        os.environ.pop("KAGYA_TEST_ADMIN_TOKEN", None)
    app_settings = settings or _settings(tmp_path)
    app = create_app(app_settings)
    app.state.model_provider = ThinkingProvider()
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
