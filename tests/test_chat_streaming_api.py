from pathlib import Path
import json
import re
from threading import Event
import time

import pytest

from kagya.operation_status import OperationState
from tests.test_fastapi_backend import ThinkingProvider, _client, _settings


class BlockingThinkingProvider(ThinkingProvider):
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()

    def stream_generate(self, prompt: str, cancellation_token=None):
        del prompt
        self.entered.set()
        self.release.wait(2)
        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()
        yield self.response_text


def _completed(client, operation_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/chat/jobs/{operation_id}")
        assert response.status_code == 200
        operation = response.json()["operation"]
        if operation["status"] in {
            OperationState.COMPLETED.value,
            OperationState.FAILED.value,
            OperationState.CANCELED.value,
        }:
            return operation
        time.sleep(0.01)
    raise AssertionError("chat job did not complete")


def test_chat_job_status_result_idempotency_and_sse_reconnect(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = {"Idempotency-Key": "same-request", "X-KAGYA-Client-ID": "client-1"}
    first = client.post(
        "/api/chat/jobs", json={"text": "hello", "attachments": []}, headers=headers
    )
    duplicate = client.post(
        "/api/chat/jobs", json={"text": "ignored", "attachments": []}, headers=headers
    )

    assert first.status_code == 202
    assert duplicate.status_code == 202
    assert duplicate.json()["duplicate"] is True
    operation_id = first.json()["operation"]["operation_id"]
    assert duplicate.json()["operation"]["operation_id"] == operation_id
    operation = _completed(client, operation_id)
    assert operation["result_available"] is True
    result = client.get(f"/api/chat/jobs/{operation_id}/result")
    assert result.status_code == 200
    assert result.json()["result"]["response"] == "Visible API answer."

    stream = client.get(f"/api/chat/jobs/{operation_id}/events")
    event_ids = [
        int(value) for value in re.findall(r"^id: (\d+)$", stream.text, re.MULTILINE)
    ]
    assert event_ids == sorted(set(event_ids))
    assert "event: token" in stream.text
    assert stream.text.count("event: final") == 1
    reconnect = client.get(
        f"/api/chat/jobs/{operation_id}/events",
        headers={"Last-Event-ID": str(event_ids[-2])},
    )
    reconnect_ids = [
        int(value) for value in re.findall(r"^id: (\d+)$", reconnect.text, re.MULTILINE)
    ]
    assert reconnect_ids == [event_ids[-1]]
    assert reconnect.text.count("event: final") == 1


def test_private_and_think_sentinels_never_reach_job_surfaces_or_store(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    client.app.state.model_provider = ThinkingProvider()
    submitted = client.post(
        "/api/chat/jobs",
        json={"text": "PRIVATE_INPUT_SENTINEL", "attachments": []},
        headers={"Idempotency-Key": "private", "X-KAGYA-Client-ID": "client"},
    )
    operation_id = submitted.json()["operation"]["operation_id"]
    _completed(client, operation_id)

    status = client.get(f"/api/chat/jobs/{operation_id}").text
    result = client.get(f"/api/chat/jobs/{operation_id}/result").text
    events = client.get(f"/api/chat/jobs/{operation_id}/events").text
    registry_path = client.app.state.settings.agent_state.path.parent / "chat_jobs.json"
    stored = registry_path.read_text()
    for surface in (status, result, events, stored):
        assert "<think>" not in surface
        assert "debug thought" not in surface
        assert "PRIVATE_INPUT_SENTINEL" not in surface


def test_running_cancel_leaves_no_raw_chroma_document(tmp_path: Path) -> None:
    client = _client(tmp_path)
    provider = BlockingThinkingProvider()
    client.app.state.model_provider = provider
    with client:
        submitted = client.post(
            "/api/chat/jobs",
            json={"text": "CANCELED_RAW_SENTINEL", "attachments": []},
            headers={"Idempotency-Key": "cancel-memory"},
        )
        operation_id = submitted.json()["operation"]["operation_id"]
        event_id = submitted.json()["operation"]["event_id"]
        assert provider.entered.wait(1)

        canceled = client.delete(f"/api/chat/jobs/{operation_id}")
        assert canceled.status_code == 200
        provider.release.set()
        assert _completed(client, operation_id)["status"] == "canceled"

        raw = client.app.state.memory_system.db1.get(include=["documents", "metadatas"])
        assert raw["ids"] == []
        assert "CANCELED_RAW_SENTINEL" not in str(raw)
        assert event_id not in str(raw)


def test_delete_after_external_prepare_boundary_returns_409_and_completes(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    entered_write = Event()
    release = Event()

    def pause_external_write(stage: str, transaction_id: str) -> None:
        del transaction_id
        if stage == "external_write":
            entered_write.set()
            release.wait(2)

    client.app.state.memory_system.set_external_boundary_injector(pause_external_write)
    with client:
        submitted = client.post(
            "/api/chat/jobs",
            json={"text": "finalizing", "attachments": []},
            headers={"Idempotency-Key": "finalizing-delete"},
        )
        operation_id = submitted.json()["operation"]["operation_id"]
        assert entered_write.wait(1)

        rejected = client.delete(f"/api/chat/jobs/{operation_id}")
        assert rejected.status_code == 409
        assert rejected.json()["detail"] == "already_finalizing"
        release.set()
        assert _completed(client, operation_id)["status"] == "completed"


@pytest.mark.parametrize(
    ("component", "stage"),
    [
        ("state_wal", "after_wal_append"),
        ("agent_state_store", "snapshot_atomic_replace"),
        ("event_journal", "before_journal_completed"),
    ],
)
def test_commit_boundary_failure_is_indeterminate_then_restart_commits(
    tmp_path: Path, component: str, stage: str
) -> None:
    settings = _settings(tmp_path)
    injected = False

    def fail_once(current: str) -> None:
        nonlocal injected
        if current == stage and not injected:
            injected = True
            raise OSError(stage)

    with _client(tmp_path, settings=settings) as client:
        target = getattr(client.app.state, component)
        target._failure_injector = fail_once
        submitted = client.post(
            "/api/chat/jobs",
            json={"text": f"recover {stage}", "attachments": []},
            headers={"Idempotency-Key": stage},
        )
        operation_id = submitted.json()["operation"]["operation_id"]
        event_id = submitted.json()["operation"]["event_id"]
        deadline = time.monotonic() + 3
        operation = None
        while time.monotonic() < deadline:
            operation = client.get(f"/api/chat/jobs/{operation_id}").json()["operation"]
            if operation["error_code"] == "commit_indeterminate":
                break
            time.sleep(0.01)
        assert operation is not None
        assert operation["status"] == "finalizing"
        assert operation["event_id"] == event_id
        assert operation["error_code"] == "commit_indeterminate"
        assert client.get("/health/ready").status_code == 503
        events = client.app.state.chat_job_registry.events_after(operation_id, 0)
        assert all(event.event not in {"token", "final"} for event in events)

    with _client(tmp_path, settings=settings) as restarted:
        operation = restarted.get(f"/api/chat/jobs/{operation_id}").json()["operation"]
        assert operation["status"] == "completed"
        assert operation["event_id"] == event_id
        result = restarted.get(f"/api/chat/jobs/{operation_id}/result")
        assert result.status_code == 200
        assert result.json()["result"]["response"] == "Visible API answer."
        stream = restarted.get(f"/api/chat/jobs/{operation_id}/events")
        event_ids = [
            int(value)
            for value in re.findall(r"^id: (\d+)$", stream.text, re.MULTILINE)
        ]
        assert event_ids == sorted(set(event_ids))
        assert stream.text.count("event: final") == 1
        records = restarted.app.state.memory_system.db1.get(
            where={"source_event_id": event_id}
        )
        assert len(records["ids"]) == 1


@pytest.mark.parametrize("source_available", [True, False])
def test_restart_reconstructs_committed_result_or_returns_bounded_409(
    tmp_path: Path, source_available: bool
) -> None:
    settings = _settings(tmp_path)
    sentinel = "PRIVATE_RECONSTRUCTION_SENTINEL"
    with _client(tmp_path, settings=settings) as client:
        submitted = client.post(
            "/api/chat/jobs",
            json={"text": sentinel, "attachments": []},
            headers={"Idempotency-Key": f"reconstruct-{source_available}"},
        )
        operation_id = submitted.json()["operation"]["operation_id"]
        event_id = submitted.json()["operation"]["event_id"]
        assert _completed(client, operation_id)["status"] == "completed"
        if not source_available:
            source = client.app.state.memory_system.db1.get(
                where={"source_event_id": event_id}
            )
            client.app.state.memory_system.db1.delete(ids=source["ids"])

    registry_path = settings.agent_state.path.parent / "chat_jobs.json"
    values = json.loads(registry_path.read_text())
    record = next(
        item for item in values if item["status"]["operation_id"] == operation_id
    )
    record["result"] = None
    record["pending_result"] = None
    record["status"].update(
        status="finalizing",
        completed_at=None,
        result_available=False,
        error_code="commit_indeterminate",
    )
    registry_path.write_text(json.dumps(values))

    with _client(tmp_path, settings=settings) as restarted:
        operation = restarted.get(f"/api/chat/jobs/{operation_id}").json()["operation"]
        assert operation["status"] == "completed"
        result = restarted.get(f"/api/chat/jobs/{operation_id}/result")
        if source_available:
            assert result.status_code == 200
            assert result.json()["result"]["response"] == "Visible API answer."
            assert sentinel not in result.text
            records = restarted.app.state.memory_system.db1.get(
                where={"source_event_id": event_id}
            )
            assert len(records["ids"]) == 1
        else:
            assert operation["error_code"] == "committed_result_unavailable"
            assert operation["result_available"] is False
            assert result.status_code == 409
            assert sentinel not in result.text
