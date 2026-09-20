import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from kagya.runtime import ContextType

from test_fastapi_backend import PRIVATE_SENTINEL, _client, admin_headers
from kagya.runtime import EventLifecycle


def _seed(client: TestClient) -> tuple[str, str]:
    registry = client.app.state.main_loop.context_registry
    first = registry.create("ctx-a", ContextType.CONVERSATION, "web")
    second = registry.create("ctx-b", ContextType.CONVERSATION, "web")
    return first.context_id, second.context_id


def test_context_api_requires_admin_and_projects_deterministically(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _seed(client)
        assert client.get("/api/contexts").status_code == 401
        response = client.get("/api/contexts", headers=admin_headers())
        assert response.status_code == 200
        assert [item["context_id"] for item in response.json()["contexts"]] == [
            "ctx-a",
            "ctx-b",
        ]
        assert set(response.json()["contexts"][0]) == {
            "context_id",
            "context_type",
            "source_channel",
            "source_session_id",
            "participant_refs",
            "parent_context_id",
            "related_context_ids",
            "status",
            "created_revision",
            "last_modified_revision",
            "started_at",
            "last_active_at",
            "is_current",
        }


def test_context_api_lifecycle_and_relation_are_runtime_mutations(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        first, second = _seed(client)
        response = client.post(
            f"/api/contexts/{first}/suspend", headers=admin_headers()
        )
        assert response.status_code == 200
        assert response.json()["status"] == "suspended"
        response = client.post(
            f"/api/contexts/{first}/relations",
            headers=admin_headers(),
            json={"related_context_id": second},
        )
        assert response.status_code == 200
        assert [item["context_id"] for item in response.json()["contexts"]] == [
            first,
            second,
        ]
        _assert_durable_context_event(client, "api.contexts.relate")
        record = client.app.state.event_journal.records[-1]
        assert record.event_type.value == "context_update"
        assert record.source.value == "api.contexts.relate"


def test_context_api_current_lifecycle_durability_and_restart_semantics(
    tmp_path: Path,
) -> None:
    settings = None
    with _client(tmp_path, settings=settings) as client:
        created = client.post(
            "/api/chat",
            json={
                "message": "create current context",
                "attachments": [],
                "client_session_id": "current-lifecycle",
            },
        )
        assert created.status_code == 200
        context_id = next(
            item["context_id"]
            for item in client.get("/api/contexts", headers=admin_headers()).json()[
                "contexts"
            ]
            if item["source_session_id"] == "current-lifecycle"
        )
        registry = client.app.state.main_loop.context_registry

        suspended = client.post(
            f"/api/contexts/{context_id}/suspend", headers=admin_headers()
        )
        assert suspended.status_code == 200
        assert suspended.json()["status"] == "suspended"
        assert suspended.json()["is_current"] is False
        assert client.get("/api/contexts", headers=admin_headers()).json()[
            "current_context_id"
        ] is None
        _assert_durable_context_event(client, "api.contexts.suspend")
        suspended_select = client.post(
            "/api/chat",
            json={
                "message": "suspended selector",
                "attachments": [],
                "context_id": context_id,
            },
        )
        assert suspended_select.status_code == 409
        assert (
            client.app.state.agent_state_store.load()
            .context_state.to_registry_state()
            == registry.state
        )

        resumed = client.post(
            f"/api/contexts/{context_id}/resume", headers=admin_headers()
        )
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "active"
        assert resumed.json()["is_current"] is False
        assert registry.current_context_id is None
        _assert_durable_context_event(client, "api.contexts.resume")

        selected = client.post(
            "/api/chat",
            json={
                "message": "select resumed context",
                "attachments": [],
                "context_id": context_id,
            },
        )
        assert selected.status_code == 200
        assert registry.current_context_id == context_id

        closed = client.post(
            f"/api/contexts/{context_id}/close", headers=admin_headers()
        )
        assert closed.status_code == 200
        assert closed.json()["status"] == "closed"
        assert closed.json()["is_current"] is False
        assert registry.current_context_id is None
        assert registry.get(context_id).status.value == "closed"
        _assert_durable_context_event(client, "api.contexts.close")
        assert (
            client.app.state.agent_state_store.load()
            .context_state.to_registry_state()
            == registry.state
        )
        committed = registry.state

    with _client(tmp_path, settings=settings) as restarted:
        restored = restarted.app.state.main_loop.context_registry.state
        assert restored == committed
        assert restored.current_context_id is None
        assert restored.frames[0].status.value == "closed"
        rejected = restarted.post(
            "/api/chat",
            json={
                "message": "closed context",
                "attachments": [],
                "context_id": context_id,
            },
        )
        assert rejected.status_code == 409


def test_context_api_active_current_pointer_survives_restart(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = client.post(
            "/api/chat",
            json={
                "message": "active restart context",
                "attachments": [],
                "client_session_id": "active-restart",
            },
        )
        assert created.status_code == 200
        before = client.app.state.main_loop.context_registry.state
        current_id = before.current_context_id
        assert current_id is not None

    with _client(tmp_path) as restarted:
        after = restarted.app.state.main_loop.context_registry.state
        assert after == before
        assert after.current_context_id == current_id
        frame = restarted.get(
            f"/api/contexts/{current_id}", headers=admin_headers()
        ).json()
        assert frame["status"] == "active"
        assert frame["is_current"] is True


def test_context_api_transitions_are_idempotent_and_close_preserves_frame(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        first, _ = _seed(client)
        headers = admin_headers()
        suspended = client.post(
            f"/api/contexts/{first}/suspend", headers=headers
        ).json()
        again = client.post(f"/api/contexts/{first}/suspend", headers=headers).json()
        assert again == suspended
        resumed = client.post(f"/api/contexts/{first}/resume", headers=headers).json()
        assert resumed["status"] == "active"
        assert resumed["is_current"] is False
        closed = client.post(f"/api/contexts/{first}/close", headers=headers).json()
        assert closed["status"] == "closed"
        assert closed["context_id"] == first
        assert (
            client.post(f"/api/contexts/{first}/close", headers=headers).json()
            == closed
        )


def test_context_api_reads_are_pure_and_unknown_is_not_found(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _seed(client)
        registry = client.app.state.main_loop.context_registry
        before_state = registry.state
        before_records = client.app.state.event_journal.records
        first = client.get("/api/contexts", headers=admin_headers()).json()
        second = client.get("/api/contexts", headers=admin_headers()).json()
        assert first == second
        assert registry.state == before_state
        assert client.app.state.event_journal.records == before_records
        assert (
            client.get("/api/contexts/missing", headers=admin_headers()).json()
            == {"detail": "Context not found"}
        )


def test_context_api_reads_preserve_all_authority_hashes(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        first, second = _seed(client)
        registry = client.app.state.main_loop.context_registry
        registry.set_current(first)
        working_memory = client.app.state.main_loop.working_memory
        before_registry = registry.state
        before_working_memory = (working_memory.revision, working_memory.items)
        before_snapshot = client.app.state.agent_state_store.load()
        before_snapshot_hash = client.app.state.agent_state_store.snapshot_hash(
            before_snapshot
        )
        before_snapshot_bytes = client.app.state.agent_state_store.path.read_bytes()
        before_memory = (
            client.app.state.memory_system.db1.get(),
            client.app.state.memory_system.db2.get(),
        )
        before_journal = client.app.state.event_journal.path.read_bytes()

        for _ in range(3):
            assert client.get("/api/contexts", headers=admin_headers()).status_code == 200
            assert (
                client.get(f"/api/contexts/{first}", headers=admin_headers()).status_code
                == 200
            )
            assert (
                client.get(f"/api/contexts/{second}", headers=admin_headers()).status_code
                == 200
            )
            registry.compatibility(first, first)
            working_memory.select_contextual(
                lambda _item: pytest.fail("read-only selection resolved an item"),
                registry,
                first,
            )

        assert registry.state == before_registry
        assert (working_memory.revision, working_memory.items) == before_working_memory
        after_snapshot = client.app.state.agent_state_store.load()
        assert client.app.state.agent_state_store.snapshot_hash(after_snapshot) == (
            before_snapshot_hash
        )
        assert client.app.state.agent_state_store.path.read_bytes() == before_snapshot_bytes
        assert (
            client.app.state.memory_system.db1.get(),
            client.app.state.memory_system.db2.get(),
        ) == before_memory
        assert client.app.state.event_journal.path.read_bytes() == before_journal


def test_context_api_privacy_keeps_raw_chat_in_memory_only(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/chat",
            json={"message": PRIVATE_SENTINEL, "attachments": []},
        )
        assert response.status_code == 200
        contexts = client.get("/api/contexts", headers=admin_headers())
        assert contexts.status_code == 200
        assert PRIVATE_SENTINEL not in contexts.text

        state_path = client.app.state.agent_state_store.path
        assert PRIVATE_SENTINEL not in state_path.read_text()
        assert PRIVATE_SENTINEL not in client.app.state.event_journal.path.read_text()
        for path in client.app.state.state_wal.root.rglob("*"):
            if path.is_file():
                assert PRIVATE_SENTINEL not in path.read_text()
        assert PRIVATE_SENTINEL in str(client.app.state.memory_system.db1.get())


def test_context_api_handler_failure_rolls_back_real_lifecycle_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        created = client.post(
            "/api/chat",
            json={
                "message": "committed context before failure",
                "attachments": [],
                "client_session_id": "failure-lifecycle",
            },
        )
        assert created.status_code == 200
        context_id = next(
            item["context_id"]
            for item in client.get("/api/contexts", headers=admin_headers()).json()[
                "contexts"
            ]
            if item["source_session_id"] == "failure-lifecycle"
        )
        registry = client.app.state.main_loop.context_registry
        before = registry.state
        original_suspend = registry.suspend

        def mutate_then_fail(value: str):
            original_suspend(value)
            raise RuntimeError(PRIVATE_SENTINEL)

        monkeypatch.setattr(registry, "suspend", mutate_then_fail)
        with pytest.raises(RuntimeError, match=PRIVATE_SENTINEL):
            client.post(
                f"/api/contexts/{context_id}/suspend", headers=admin_headers()
            )

        assert registry.state == before
        context_events = [
            record
            for record in client.app.state.event_journal.records
            if record.source is not None
            and record.source.value == "api.contexts.suspend"
        ]
        assert context_events[-1].lifecycle is EventLifecycle.FAILED
        assert not any(
            record.lifecycle is EventLifecycle.COMPLETED for record in context_events
        )
        assert PRIVATE_SENTINEL not in client.app.state.event_journal.path.read_text()


def test_context_and_internal_state_roll_back_on_preparation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        registry = client.app.state.main_loop.context_registry
        working_memory = client.app.state.main_loop.working_memory
        before_context = registry.state
        before_working_memory = (working_memory.revision, working_memory.items)
        before_emotion = client.app.state.main_loop.emotion_engine.state

        def fail_preparation(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("preparation unavailable")

        monkeypatch.setattr(
            client.app.state.transaction_coordinator,
            "prepare_result",
            fail_preparation,
        )
        response = client.post(
            "/api/chat", json={"message": "preparation failure", "attachments": []}
        )

        assert response.status_code == 500
        assert response.json() == {"detail": "Agent mutation durability is indeterminate"}
        assert registry.state == before_context
        assert (working_memory.revision, working_memory.items) == before_working_memory
        assert client.app.state.main_loop.emotion_engine.state == before_emotion
        assert client.app.state.agent_state_store.load().context_state.to_registry_state() == (
            before_context
        )
        chat_events = [
            record
            for record in client.app.state.event_journal.records
            if record.source is not None and record.source.value == "api.chat"
        ]
        assert chat_events[-1].lifecycle is EventLifecycle.FAILED
        assert not any(
            record.lifecycle is EventLifecycle.COMPLETED for record in chat_events
        )


def test_context_api_invalid_ids_are_bounded_and_not_admitted(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _seed(client)
        before = client.app.state.event_journal.records
        response = client.post(
            "/api/contexts/../suspend", headers=admin_headers()
        )
        assert response.status_code in {404, 422}
        response = client.post(
            "/api/contexts/bad%20value/suspend", headers=admin_headers()
        )
        assert response.status_code == 422
        assert "bad value" not in response.text
        response = client.post(
            "/api/contexts/ctx-a/relations",
            headers=admin_headers(),
            json={"related_context_id": "bad value"},
        )
        assert response.status_code == 422
        assert "bad value" not in response.text
        assert client.app.state.event_journal.records == before


def _assert_durable_context_event(client: TestClient, source: str) -> None:
    records = [
        record
        for record in client.app.state.event_journal.records
        if record.source is not None and record.source.value == source
    ]
    assert [record.lifecycle for record in records] == [
        EventLifecycle.ACCEPTED,
        EventLifecycle.STARTED,
        EventLifecycle.PREPARED,
        EventLifecycle.COMPLETED,
    ]
