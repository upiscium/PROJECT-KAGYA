from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from project_kagya.multimodal_fastapi_interface import create_app


def test_ingest_accepts_text_only() -> None:
    client = TestClient(create_app())

    response = client.post("/ingest", data={"text": "hello"})

    assert response.status_code == 200
    assert response.json()["received_text"] == "hello"
    assert response.json()["modalities"] == ["text"]


def test_ingest_accepts_image_upload() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/ingest",
        data={"text": "hello"},
        files={"image": ("image.png", BytesIO(b"fake"), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["modalities"] == ["text", "image"]


def test_ingest_rejects_empty_input() -> None:
    client = TestClient(create_app())

    response = client.post("/ingest", data={})

    assert response.status_code == 400


def test_chat_accepts_message() -> None:
    client = TestClient(create_app())

    response = client.post("/chat", data={"message": "hello"})

    assert response.status_code == 200
    assert response.json()["message"] == "hello"


def test_chat_uses_injected_backend() -> None:
    class DummyBackend:
        def reply(self, message: str) -> str:
            return f"response: {message}"

    client = TestClient(create_app(DummyBackend()))

    response = client.post("/chat", data={"message": "hello"})

    assert response.status_code == 200
    assert response.json()["message"] == "response: hello"


def test_stream_echoes_messages() -> None:
    client = TestClient(create_app())

    with client.websocket_connect("/stream") as websocket:
        websocket.send_text("hello")
        message = websocket.receive_json()

    assert message == {"status": "ok", "echo": "hello"}
