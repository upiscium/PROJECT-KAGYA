from pathlib import Path
import re
import time

from kagya.operation_status import OperationState
from tests.test_fastapi_backend import ThinkingProvider, _client


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
    event_ids = [int(value) for value in re.findall(r"^id: (\d+)$", stream.text, re.MULTILINE)]
    assert event_ids == sorted(set(event_ids))
    assert "event: token" in stream.text
    assert stream.text.count("event: final") == 1
    reconnect = client.get(
        f"/api/chat/jobs/{operation_id}/events",
        headers={"Last-Event-ID": str(event_ids[-2])},
    )
    reconnect_ids = [
        int(value)
        for value in re.findall(r"^id: (\d+)$", reconnect.text, re.MULTILINE)
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
