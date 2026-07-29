from __future__ import annotations

import base64
from dataclasses import replace
from datetime import UTC, datetime
import importlib
from pathlib import Path
from threading import Event, Lock, Thread
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kagya.api.server import _consume_interrupted_sequence, create_app
from kagya.config import Settings, load_settings
from kagya.config.schema import EtcdFailoverSettings
from kagya.failover import (
    EtcdFencingAuthority,
    FailoverError,
    FailoverManifest,
    NotAuthoritativeError,
    core_files,
    restore_and_preflight,
)
from kagya.memory.dual_memory_system import configured_memory_identity
from kagya.runtime import (
    AgentEvent,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeJournalError,
    AgentRuntimeQueueFull,
    EventJournal,
    StateWAL,
    hash_snapshot,
)
from kagya.operation_status import OperationCancelCode
from kagya.runtime.agent_state import AgentStateStore, default_agent_state_snapshot
from tests.chat_job_helpers import ChatJobRegistry


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class FakeEtcd:
    def __init__(self) -> None:
        self.revision = 1
        self.next_lease = 100
        self.values: dict[bytes, tuple[bytes, int, int, int]] = {}
        self.leases: set[int] = set()
        self.partitioned = False

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.partitioned:
            raise FailoverError("partition")
        if path == "/v3/lease/grant":
            self.next_lease += 1
            self.leases.add(self.next_lease)
            return {"ID": str(self.next_lease), "TTL": payload["TTL"]}
        if path == "/v3/lease/keepalive":
            lease = int(payload["ID"])
            return {
                "result": {"ID": str(lease), "TTL": 15 if lease in self.leases else 0}
            }
        if path == "/v3/lease/revoke":
            self.expire(int(payload["ID"]))
            return {}
        if path == "/v3/kv/range":
            item = self.values.get(_decode(payload["key"]))
            if item is None:
                return {"header": {"revision": str(self.revision)}, "kvs": []}
            value, create, modified, lease = item
            return {
                "header": {"revision": str(self.revision)},
                "kvs": [
                    {
                        "value": _encode(value),
                        "create_revision": str(create),
                        "mod_revision": str(modified),
                        "lease": str(lease),
                    }
                ],
            }
        if path == "/v3/kv/deleterange":
            key = _decode(payload["key"])
            if key in self.values:
                self.revision += 1
                del self.values[key]
            return {"header": {"revision": str(self.revision)}, "deleted": "1"}
        if path != "/v3/kv/txn":
            raise AssertionError(path)
        succeeded = all(self._compare(item) for item in payload["compare"])
        operations = payload["success"] if succeeded else payload["failure"]
        if operations:
            self.revision += 1
            for operation in operations:
                delete = operation.get("request_delete_range")
                if delete is not None:
                    self.values.pop(_decode(delete["key"]), None)
                    continue
                put = operation["request_put"]
                key = _decode(put["key"])
                value = _decode(put["value"])
                lease = int(put.get("lease", 0))
                previous = self.values.get(key)
                created = self.revision if previous is None else previous[1]
                self.values[key] = (value, created, self.revision, lease)
        return {"succeeded": succeeded, "header": {"revision": str(self.revision)}}

    def expire(self, lease: int) -> None:
        self.leases.discard(lease)
        removed = [key for key, value in self.values.items() if value[3] == lease]
        if removed:
            self.revision += 1
        for key in removed:
            del self.values[key]

    def _compare(self, compare: dict[str, str]) -> bool:
        key = _decode(compare["key"])
        item = self.values.get(key)
        target = compare["target"]
        if target == "CREATE":
            actual: str = "0" if item is None else str(item[1])
            expected = compare["create_revision"]
        elif target == "MOD":
            actual = "0" if item is None else str(item[2])
            expected = compare["mod_revision"]
        else:
            actual = _encode(b"" if item is None else item[0])
            expected = compare["value"]
        return actual == expected


def test_etcd_election_has_one_winner_and_higher_epoch_after_expiry() -> None:
    etcd = FakeEtcd()
    first = _authority("node-a", etcd)
    contender = _authority("node-b", etcd)

    assert first.acquire() is True
    first_token = first.fencing_token
    assert contender.acquire() is False
    assert contender.acquisition_conflicts == 1
    assert first._lease_id is not None
    etcd.expire(first._lease_id)
    assert contender.acquire() is True

    assert contender.fencing_token > first_token


def test_failover_config_defaults_and_invalid_combinations_are_exact() -> None:
    settings = load_settings(CONFIG_PATH)
    assert settings.failover.enabled is False
    assert settings.failover.subject_role.value == "active"
    assert settings.failover.etcd.key_prefix == "/kagya/subject"

    raw = settings.model_dump(mode="python")
    raw["failover"]["automatic_promotion"] = True
    with pytest.raises(ValueError, match="only valid for standby"):
        Settings.model_validate(raw)
    raw = settings.model_dump(mode="python")
    raw["failover"]["etcd"]["renew_interval_seconds"] = 8
    with pytest.raises(ValueError, match="less than half"):
        Settings.model_validate(raw)


def test_renewal_is_stable_and_partition_fails_closed() -> None:
    etcd = FakeEtcd()
    authority = _authority("node-a", etcd)
    assert authority.acquire()

    authority.renew_once()
    authority.renew_once()
    token = authority.fencing_token
    etcd.partitioned = True

    with pytest.raises(FailoverError):
        authority.renew_once()
    assert authority.is_authoritative is False
    assert authority._fencing_token == token


def test_automatic_campaign_promotes_after_old_lease_expires() -> None:
    etcd = FakeEtcd()
    active = _authority("node-a", etcd)
    standby = _authority("node-b", etcd)
    assert active.acquire()
    promoted = Event()
    standby.start_campaign(promoted.set)
    assert active._lease_id is not None
    etcd.expire(active._lease_id)

    assert promoted.wait(2)
    assert standby.is_authoritative
    assert standby.fencing_token > active.fencing_token
    standby.close()


def test_stale_authority_cannot_publish_after_promotion() -> None:
    etcd = FakeEtcd()
    stale = _authority("node-a", etcd)
    promoted = _authority("node-b", etcd)
    assert stale.acquire()
    assert stale._lease_id is not None
    etcd.expire(stale._lease_id)
    assert promoted.acquire()

    with pytest.raises(NotAuthoritativeError, match="not_authoritative"):
        stale.publish(_manifest(stale), {"snapshot": b"stale"})
    assert stale.stale_write_rejections == 1

    watermark_key = b"/test/subject/watermark"
    assert watermark_key not in etcd.values


def test_terminal_ack_waits_for_replica_publication() -> None:
    order: list[str] = []

    class Journal:
        def accepted(self, event: AgentEvent) -> None:
            order.append("accepted")

        def started(self, event: AgentEvent) -> None:
            order.append("started")

        def completed(self, event: AgentEvent, snapshot_hash: str) -> None:
            order.append("journal_terminal")

        def failed(
            self, event: AgentEvent, category: str, snapshot_hash: str | None
        ) -> None:
            order.append("journal_failed")

    entered = Event()
    release = Event()

    def publish(event: AgentEvent, snapshot_hash: str) -> None:
        order.append("publish_entered")
        entered.set()
        release.wait(2)
        order.append("watermark_published")

    runtime = AgentRuntime(
        queue_capacity=1,
        event_journal=Journal(),
        completion_hook=lambda event: "a" * 64,
        terminal_hook=publish,
    )
    runtime.start()
    future = runtime.submit(AgentEventType.CHAT, source="test", handler=lambda: "ok")
    assert entered.wait(2)
    assert future.done() is False
    release.set()
    assert future.result(timeout=2).value == "ok"
    runtime.shutdown()
    assert order.index("journal_terminal") < order.index("watermark_published")


def test_acceptance_ack_waits_for_replica_publication() -> None:
    entered = Event()
    release = Event()

    class Journal:
        def accepted(self, event: AgentEvent) -> None:
            pass

        def started(self, event: AgentEvent) -> None:
            pass

        def completed(self, event: AgentEvent, snapshot_hash: str) -> None:
            pass

        def failed(
            self, event: AgentEvent, category: str, snapshot_hash: str | None
        ) -> None:
            pass

    def replicate(_event: AgentEvent) -> None:
        entered.set()
        release.wait(2)

    runtime = AgentRuntime(
        queue_capacity=1,
        event_journal=Journal(),
        admission_hook=replicate,
        completion_hook=lambda event: "a" * 64,
    )
    runtime.start()
    submitted = Event()

    def submit() -> None:
        runtime.submit(AgentEventType.CHAT, source="test", handler=lambda: "ok")
        submitted.set()

    thread = Thread(target=submit)
    thread.start()
    assert entered.wait(2)
    assert submitted.is_set() is False
    release.set()
    thread.join(timeout=2)
    assert submitted.is_set()
    runtime.shutdown()


def test_handler_waits_for_started_evidence_publication() -> None:
    started_publication = Event()
    release = Event()
    handler_entered = Event()

    def publish_started(_event: AgentEvent) -> None:
        started_publication.set()
        release.wait(2)

    runtime = AgentRuntime(queue_capacity=1, started_hook=publish_started)
    runtime.start()
    future = runtime.submit(
        AgentEventType.CHAT,
        source="test",
        handler=lambda: handler_entered.set(),
    )

    assert started_publication.wait(2)
    assert handler_entered.is_set() is False
    release.set()
    future.result(timeout=2)
    assert handler_entered.is_set()
    runtime.shutdown()


def test_chat_enqueue_never_holds_registry_lock_while_admission_waits(
    tmp_path: Path,
) -> None:
    boundary = Lock()
    first_handler_entered = Event()
    second_admission_entered = Event()
    handler_acquired_registry = Event()

    def admission(event: AgentEvent) -> None:
        if event.source == "api.chat.job":
            second_admission_entered.set()
        with boundary:
            pass

    runtime = AgentRuntime(
        queue_capacity=2,
        admission_hook=admission,
        event_boundary_lock=boundary,
        event_boundary_types={AgentEventType.ADAPTER_UPDATE},
    )
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        lambda request: {"response": str(request["text"])},
    )

    def first_handler() -> None:
        first_handler_entered.set()
        assert second_admission_entered.wait(2)
        with registry._lock:
            handler_acquired_registry.set()

    first = runtime.submit(
        AgentEventType.MEMORY_READ, source="test", handler=first_handler
    )
    assert first_handler_entered.wait(2)
    enqueue_finished = Event()

    def enqueue() -> None:
        registry.enqueue(
            {"text": "second"},
            client_id="client",
            idempotency_key="key",
            correlation_id="correlation",
        )
        enqueue_finished.set()

    thread = Thread(target=enqueue)
    thread.start()
    assert handler_acquired_registry.wait(2)
    first.result(timeout=2)
    thread.join(timeout=2)
    assert enqueue_finished.is_set()
    registry.shutdown()
    runtime.shutdown()
    registry.close()


def test_event_boundary_lock_is_restricted_to_adapter_updates() -> None:
    boundary = Lock()
    boundary.acquire()
    runtime = AgentRuntime(
        queue_capacity=2,
        event_boundary_lock=boundary,
        event_boundary_types={AgentEventType.ADAPTER_UPDATE},
    )
    runtime.start()
    chat = runtime.submit(AgentEventType.CHAT, source="test", handler=lambda: "chat")
    assert chat.result(timeout=2).value == "chat"
    adapter_entered = Event()
    adapter = runtime.submit(
        AgentEventType.ADAPTER_UPDATE,
        source="test",
        handler=lambda: adapter_entered.set(),
    )
    assert adapter_entered.wait(0.1) is False
    boundary.release()
    assert adapter.result(timeout=2).value is None
    runtime.shutdown()


def test_chat_queue_full_cleanup_publishes_outside_registry_lock(
    tmp_path: Path,
) -> None:
    release = Event()
    entered = Event()
    runtime = AgentRuntime(queue_capacity=1)
    runtime.start()

    def blocking() -> None:
        entered.set()
        release.wait(2)

    active = runtime.submit(AgentEventType.MEMORY_READ, source="test", handler=blocking)
    assert entered.wait(2)
    queued = runtime.submit(
        AgentEventType.MEMORY_READ, source="test", handler=lambda: None
    )
    registry: ChatJobRegistry

    def persistence_hook() -> None:
        acquired = Event()

        def acquire() -> None:
            if registry._lock.acquire(timeout=1):
                acquired.set()
                registry._lock.release()

        thread = Thread(target=acquire)
        thread.start()
        thread.join(timeout=2)
        assert acquired.is_set()

    registry = ChatJobRegistry(
        tmp_path / "queue-full.json",
        runtime,
        lambda request: {"response": str(request["text"])},
        persistence_hook=persistence_hook,
    )
    with pytest.raises(AgentRuntimeQueueFull):
        registry.enqueue(
            {"text": "full"},
            client_id="client",
            idempotency_key="full",
            correlation_id="correlation",
        )
    release.set()
    active.result(timeout=2)
    queued.result(timeout=2)
    registry.shutdown()
    runtime.shutdown()
    registry.close()


def test_chat_cancel_publishes_outside_registry_lock(tmp_path: Path) -> None:
    release = Event()
    entered = Event()
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()

    def blocking() -> None:
        entered.set()
        release.wait(2)

    active = runtime.submit(AgentEventType.MEMORY_READ, source="test", handler=blocking)
    assert entered.wait(2)
    registry = ChatJobRegistry(
        tmp_path / "cancel.json",
        runtime,
        lambda request: {"response": str(request["text"])},
    )
    record, _ = registry.enqueue(
        {"text": "cancel"},
        client_id="client",
        idempotency_key="cancel",
        correlation_id="correlation",
    )
    acquired = Event()

    def persistence_hook() -> None:
        def acquire() -> None:
            if registry._lock.acquire(timeout=1):
                acquired.set()
                registry._lock.release()

        thread = Thread(target=acquire)
        thread.start()
        thread.join(timeout=2)

    registry._persistence_hook = persistence_hook
    assert (
        registry.cancel(record.status.operation_id, OperationCancelCode.CLIENT_REQUEST)
        == "canceled"
    )
    assert acquired.is_set()
    release.set()
    active.result(timeout=2)
    registry.shutdown()
    runtime.shutdown()
    registry.close()


def test_duplicate_chat_waits_for_first_admission_result(tmp_path: Path) -> None:
    admission_entered = Event()
    release_admission = Event()

    def admission(_event: AgentEvent) -> None:
        admission_entered.set()
        release_admission.wait(2)

    runtime = AgentRuntime(queue_capacity=2, admission_hook=admission)
    runtime.start()
    registry = ChatJobRegistry(
        tmp_path / "jobs.json",
        runtime,
        lambda request: {"response": str(request["text"])},
    )
    results: list[tuple[object, bool]] = []

    def enqueue() -> None:
        results.append(
            registry.enqueue(
                {"text": "same"},
                client_id="client",
                idempotency_key="same-key",
                correlation_id="correlation",
            )
        )

    first = Thread(target=enqueue)
    duplicate = Thread(target=enqueue)
    first.start()
    assert admission_entered.wait(2)
    duplicate.start()
    duplicate.join(timeout=0.1)
    assert duplicate.is_alive()
    release_admission.set()
    first.join(timeout=2)
    duplicate.join(timeout=2)

    assert sorted(created for _record, created in results) == [False, True]
    registry.shutdown()
    runtime.shutdown()
    registry.close()


def test_failed_promotion_relinquishes_then_clears_graph_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = importlib.import_module("kagya.api.server")
    settings = _settings(tmp_path).model_copy(
        update={
            "failover": _settings(tmp_path).failover.model_copy(
                update={"bootstrap_from_local_state": True}
            )
        }
    )
    order: list[str] = []

    class Authority:
        is_authoritative = True

        def start_renewal(self, _callback: object) -> None:
            order.append("renewal")

        def latest_bundle(self) -> None:
            return None

        def relinquish(self) -> None:
            order.append("relinquish")
            self.is_authoritative = False

        def assert_authoritative(self) -> None:
            if not self.is_authoritative:
                raise NotAuthoritativeError("not_authoritative")

    class PartialRuntime:
        def abort(self) -> None:
            order.append("abort")

        def shutdown(self) -> None:
            order.append("shutdown")

    authority = Authority()
    app = SimpleNamespace(
        state=SimpleNamespace(
            fencing_authority=authority,
            subject_role="standby",
            agent_runtime=None,
        )
    )
    attempts = 0

    def build(subject_app: Any, _settings: Settings, **_kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            subject_app.state.agent_runtime = PartialRuntime()
            raise RuntimeError("partial build")

    monkeypatch.setattr(server, "_build_subject_runtime", build)
    monkeypatch.setattr(server, "_publish_subject_generation", lambda *_args: None)
    monkeypatch.setattr(server, "_activate_subject_runtime", lambda *_args: None)

    with pytest.raises(RuntimeError, match="partial build"):
        server._promote_subject(app, settings, authority)

    assert order.index("relinquish") < order.index("shutdown")
    assert app.state.agent_runtime is None
    authority.is_authoritative = True
    server._promote_subject(app, settings, authority)
    assert attempts == 2
    assert app.state.subject_role == "active"


def test_publication_failure_never_returns_success() -> None:
    class Journal:
        def accepted(self, event: AgentEvent) -> None:
            pass

        def started(self, event: AgentEvent) -> None:
            pass

        def completed(self, event: AgentEvent, snapshot_hash: str) -> None:
            pass

        def failed(
            self, event: AgentEvent, category: str, snapshot_hash: str | None
        ) -> None:
            pass

    runtime = AgentRuntime(
        queue_capacity=1,
        event_journal=Journal(),
        completion_hook=lambda event: "a" * 64,
        terminal_hook=lambda event, snapshot_hash: (_ for _ in ()).throw(
            FailoverError("publish failed")
        ),
    )
    runtime.start()

    with pytest.raises(AgentRuntimeJournalError):
        runtime.execute(
            AgentEventType.CHAT, source="test", handler=lambda: "not-visible"
        )
    runtime.shutdown()


def test_promotion_preflight_restores_contiguous_sequence_and_rejects_revision(
    tmp_path: Path,
) -> None:
    source = _settings(tmp_path / "source")
    _write_sequence_one(source)
    etcd = FakeEtcd()
    authority = _authority("node-a", etcd)
    assert authority.acquire()
    manifest = _core_manifest(source, authority)
    authority.publish(manifest, core_files(source))
    bundle = authority.latest_bundle()
    assert bundle is not None

    destination = _settings(tmp_path / "destination")
    stale = default_agent_state_snapshot(1.0).model_copy(
        update={"last_processed_event_sequence": 9}
    )
    AgentStateStore(destination.agent_state.path).save(stale)
    restore_and_preflight(destination, bundle)
    restored = AgentStateStore(destination.agent_state.path).load(1.0)
    assert restored.last_processed_event_sequence == 1
    assert StateWAL(destination.agent_state_wal.path).reconstruct().sequence == 1

    raw = destination.model_dump(mode="python")
    raw["model"]["processor_revision"] = "different-processor"
    mismatched = Settings.model_validate(raw)
    with pytest.raises(FailoverError, match="model or shared memory"):
        restore_and_preflight(mismatched, bundle)

    raw = destination.model_dump(mode="python")
    raw["memory"]["embedding_revision"] = "b" * 40
    embedding_mismatched = Settings.model_validate(raw)
    with pytest.raises(FailoverError, match="model or shared memory"):
        restore_and_preflight(embedding_mismatched, bundle)


def test_interrupted_started_sequence_is_consumed_without_replaying_handler(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    snapshot = default_agent_state_snapshot(1.0)
    store = AgentStateStore(settings.agent_state.path)
    store.save(snapshot)
    wal = StateWAL(settings.agent_state_wal.path)
    wal.bootstrap(snapshot)
    journal = EventJournal(settings.agent_journal.path)
    interrupted = replace(_event(), processing_sequence=1)
    journal.started(interrupted)
    recovery = journal.reconcile(snapshot)
    app = SimpleNamespace(
        state=SimpleNamespace(
            agent_state_store=store,
            state_wal=wal,
            event_journal=journal,
        )
    )

    recovered = _consume_interrupted_sequence(app, snapshot, recovery)

    assert recovered.last_processed_event_sequence == 1
    assert wal.reconstruct().sequence == 1
    assert journal.reconcile(recovered) == []
    next_event = replace(_event(), event_id="event-2", processing_sequence=2)
    journal.started(next_event)
    journal.verify()


def test_configured_standby_is_read_only_and_builds_no_subject(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    raw = settings.model_dump(mode="python")
    raw["failover"] = {
        "enabled": True,
        "subject_role": "standby",
        "automatic_promotion": False,
        "etcd": {"endpoints": ["http://unused.test:2379"]},
    }
    raw["at_rest"]["live"]["enabled"] = True
    raw["memory"]["backend"] = "http"
    raw["memory"]["embedding_revision"] = "a" * 40
    raw["model"]["revision"] = "immutable-model"
    raw["model"]["processor_revision"] = "immutable-processor"
    raw["model"]["fallback_revision"] = "immutable-fallback"
    app = create_app(Settings.model_validate(raw))

    with TestClient(app) as client:
        response = client.post("/api/chat", json={"text": "blocked", "attachments": []})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "not_authoritative"
        readiness = client.get("/health/ready")
        assert readiness.status_code == 503
        assert readiness.json()["subject_role"] == "standby"
        assert not hasattr(app.state, "main_loop")
        assert not hasattr(app.state, "model_provider")
        assert not hasattr(app.state, "agent_runtime")


def test_failover_rejects_plaintext_state_and_mutable_revisions() -> None:
    settings = load_settings(CONFIG_PATH)
    raw = settings.model_dump(mode="python")
    raw["failover"]["enabled"] = True
    with pytest.raises(ValueError, match="live authoritative encryption"):
        Settings.model_validate(raw)

    raw["at_rest"]["live"]["enabled"] = True
    raw["memory"]["backend"] = "http"
    with pytest.raises(ValueError, match="exact immutable revisions"):
        Settings.model_validate(raw)

    raw["model"]["revision"] = "a" * 40
    raw["model"]["processor_revision"] = "b" * 40
    raw["model"]["fallback_revision"] = "c" * 40
    with pytest.raises(ValueError, match="memory embedding revision"):
        Settings.model_validate(raw)


def test_replica_watermark_cannot_move_backward() -> None:
    etcd = FakeEtcd()
    authority = _authority("node-a", etcd)
    assert authority.acquire()
    current = _manifest(authority).model_copy(update={"processing_sequence": 1})
    authority.publish(current, {"snapshot": b"current"})
    older = _manifest(authority)

    with pytest.raises(FailoverError, match="cannot move backward"):
        authority.publish(older, {"snapshot": b"older"})


def _authority(node: str, etcd: FakeEtcd) -> EtcdFencingAuthority:
    return EtcdFencingAuthority(
        node,
        EtcdFailoverSettings(
            endpoints=["http://unused.test:2379"],
            key_prefix="/test/subject",
            lease_ttl_seconds=15,
            renew_interval_seconds=1,
        ),
        etcd,
    )


def _manifest(authority: EtcdFencingAuthority) -> FailoverManifest:
    return FailoverManifest(
        node_id=authority.node_id,
        fencing_token=authority.fencing_token,
        captured_at=datetime.now(UTC),
        processing_sequence=0,
        snapshot_hash="a" * 64,
        wal_record_hash="b" * 64,
        model_id="model",
        model_revision="model-revision",
        processor_revision="processor-revision",
        fallback_id="fallback",
        fallback_revision="fallback-revision",
        memory_host="memory.test",
        memory_port=8001,
        memory_ssl=True,
        memory_tenant="tenant",
        memory_database="database",
        memory_db1_collection="db1",
        memory_db2_collection="db2",
        memory_embedding_identity="embedding",
        memory_hash="c" * 64,
        memory_transaction_head="d" * 64,
    )


def _settings(root: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": root / "chroma",
                    "db1_collection": "failover_db1",
                    "db2_collection": "failover_db2",
                }
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": root / "adapter_registry.json",
                    "eval_result_dir": root / "eval_results",
                    "eval_sets": [],
                }
            ),
            "agent_state": settings.agent_state.model_copy(
                update={"path": root / "state.json"}
            ),
            "agent_journal": settings.agent_journal.model_copy(
                update={"path": root / "journal.jsonl"}
            ),
            "agent_state_wal": settings.agent_state_wal.model_copy(
                update={"path": root / "private" / "wal.jsonl"}
            ),
            "observability": settings.observability.model_copy(
                update={
                    "metrics_path": root / "metrics.json",
                    "traces_path": root / "traces.json",
                }
            ),
        }
    )


def _write_sequence_one(settings: Settings) -> None:
    before = default_agent_state_snapshot(1.0)
    store = AgentStateStore(settings.agent_state.path)
    store.save(before)
    wal = StateWAL(settings.agent_state_wal.path)
    wal.bootstrap(before)
    journal = EventJournal(settings.agent_journal.path)
    event = replace(_event(), processing_sequence=1)
    after = before.model_copy(
        update={"saved_at": datetime.now(UTC), "last_processed_event_sequence": 1}
    )
    journal.started(event)
    journal.prepared(
        event,
        state_hash_before=hash_snapshot(before),
        state_hash_after=hash_snapshot(after),
    )
    wal.append_transition(event, before, after)
    store.save(after)
    journal.completed(event, hash_snapshot(after))


def _core_manifest(
    settings: Settings, authority: EtcdFencingAuthority
) -> FailoverManifest:
    snapshot = AgentStateStore(settings.agent_state.path).load(1.0)
    wal = StateWAL(settings.agent_state_wal.path).verify()
    journal = EventJournal(settings.agent_journal.path).verify()
    db1, db2, embedding = configured_memory_identity(settings)
    return FailoverManifest(
        node_id=authority.node_id,
        fencing_token=authority.fencing_token,
        captured_at=datetime.now(UTC),
        processing_sequence=snapshot.last_processed_event_sequence,
        snapshot_hash=hash_snapshot(snapshot),
        wal_record_hash=wal[-1].record_hash,
        journal_record_hash=journal[-1].record_hash,
        model_id=settings.model.primary_id,
        model_revision=settings.model.revision,
        processor_revision=settings.model.processor_revision,
        fallback_id=settings.model.fallback_id,
        fallback_revision=settings.model.fallback_revision,
        memory_host=settings.memory.http_host,
        memory_port=settings.memory.http_port,
        memory_ssl=settings.memory.http_ssl,
        memory_tenant=settings.memory.http_tenant,
        memory_database=settings.memory.http_database,
        memory_db1_collection=db1,
        memory_db2_collection=db2,
        memory_embedding_identity=embedding,
        memory_hash="c" * 64,
        memory_transaction_head="d" * 64,
    )


def _event() -> AgentEvent:
    now = datetime.now(UTC)
    return AgentEvent(
        event_id="event-1",
        event_type=AgentEventType.CHAT,
        source="test",
        observed_at=now,
        requested_at=now,
    )


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode(value: str) -> bytes:
    return base64.b64decode(value)
