from __future__ import annotations

import base64
import hashlib
import json
from contextlib import closing
import os
import socket
from urllib.request import Request, urlopen

from project_kagya.api import ChatService, make_handler, parse_chat_request


def _websocket_accept(key: str) -> str:
    accept_source = (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("utf-8")
    return base64.b64encode(hashlib.sha1(accept_source).digest()).decode("ascii")


def _write_ws_frame(sock: socket.socket, payload: str) -> None:
    data = payload.encode("utf-8")
    frame = bytearray([0x81])
    length = len(data)
    if length < 126:
        frame.append(0x80 | length)
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(length.to_bytes(2, "big"))
    else:
        frame.append(0x80 | 127)
        frame.extend(length.to_bytes(8, "big"))
    mask = os.urandom(4)
    frame.extend(mask)
    frame.extend(byte ^ mask[index % 4] for index, byte in enumerate(data))
    sock.sendall(frame)


def _read_ws_frame(sock: socket.socket) -> str:
    first = sock.recv(2)
    if not first:
        raise ConnectionError("closed")
    length = first[1] & 0x7F
    if length == 126:
        length = int.from_bytes(sock.recv(2), "big")
    elif length == 127:
        length = int.from_bytes(sock.recv(8), "big")
    payload = bytearray()
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("closed")
        payload.extend(chunk)
    return payload.decode("utf-8")


def test_parse_chat_request_supports_multiple_attachments(tmp_path) -> None:
    image = tmp_path / "image.png"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"png")
    audio.write_bytes(b"wav")

    request = parse_chat_request(
        {
            "text": "describe these",
            "attachments": [
                {"path": str(image)},
                {"path": str(audio)},
            ],
        }
    )

    assert request.text == "describe these"
    assert [attachment.media_type for attachment in request.attachments] == [
        "image",
        "audio",
    ]


def test_parse_chat_request_rejects_empty_payload() -> None:
    try:
        parse_chat_request({})
    except ValueError as error:
        assert str(error) == "A request must include text or attachments."
    else:
        raise AssertionError("expected ValueError")


def test_chat_handler_returns_reply(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")

    calls: dict[str, object] = {}

    class BatchFeature(dict):
        def to(self, device):
            calls["device"] = device
            return self

    class FakeProcessor:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ):
            calls["messages"] = messages
            import torch

            return BatchFeature({"input_ids": torch.tensor([[1, 2, 3]])})

        def batch_decode(self, outputs, skip_special_tokens=False):
            calls["decoded_outputs"] = outputs
            return ["assistant reply"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            import torch

            return torch.tensor([[1, 2, 3, 4, 5]])

    service = ChatService(
        model=FakeModel(),
        processor=FakeProcessor(),
        system_prompt="",
        prompt_style="auto",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
    )
    handler = make_handler(service)

    from http.server import ThreadingHTTPServer
    from threading import Thread

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            request = Request(
                f"http://{host}:{port}/chat",
                data=json.dumps(
                    {
                        "text": "describe these files",
                        "attachments": [{"path": str(image)}],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with closing(urlopen(request)) as response:
                body = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            thread.join(timeout=1)

    assert body == {"reply": "assistant reply"}
    assert calls["messages"][0]["role"] == "user"
    assert calls["messages"][0]["content"][0]["type"] == "image"
    assert calls["messages"][0]["content"][1] == {
        "type": "text",
        "text": "describe these files",
    }


def test_websocket_chat_returns_reply(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")

    class BatchFeature(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ):
            import torch

            return BatchFeature({"input_ids": torch.tensor([[1, 2, 3]])})

        def batch_decode(self, outputs, skip_special_tokens=False):
            return ["ws reply"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            import torch

            return torch.tensor([[1, 2, 3, 4, 5]])

    service = ChatService(
        model=FakeModel(),
        processor=FakeProcessor(),
        system_prompt="",
        prompt_style="auto",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
    )
    handler = make_handler(service)

    from http.server import ThreadingHTTPServer
    from threading import Thread

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            key = base64.b64encode(b"test-websocket-key").decode("ascii")
            with socket.create_connection((host, port)) as sock:
                request = (
                    f"GET /ws HTTP/1.1\r\n"
                    f"Host: {host}:{port}\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {key}\r\n"
                    "Sec-WebSocket-Version: 13\r\n\r\n"
                )
                sock.sendall(request.encode("ascii"))
                response = sock.recv(1024).decode("ascii", errors="ignore")
                assert "101 Switching Protocols" in response
                assert f"Sec-WebSocket-Accept: {_websocket_accept(key)}" in response

                _write_ws_frame(
                    sock,
                    json.dumps(
                        {
                            "text": "describe",
                            "attachments": [{"path": str(image)}],
                        }
                    ),
                )
                reply = json.loads(_read_ws_frame(sock))
        finally:
            server.shutdown()
            thread.join(timeout=1)

    assert reply == {"reply": "ws reply"}


def test_websocket_chat_accepts_inline_attachment(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")

    class BatchFeature(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ):
            import torch

            return BatchFeature({"input_ids": torch.tensor([[1, 2, 3]])})

        def batch_decode(self, outputs, skip_special_tokens=False):
            return ["inline ws reply"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            import torch

            return torch.tensor([[1, 2, 3, 4, 5]])

    service = ChatService(
        model=FakeModel(),
        processor=FakeProcessor(),
        system_prompt="",
        prompt_style="auto",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
    )
    handler = make_handler(service)

    from http.server import ThreadingHTTPServer
    from threading import Thread

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            key = base64.b64encode(b"inline-websocket-key").decode("ascii")
            with socket.create_connection((host, port)) as sock:
                request = (
                    f"GET /ws HTTP/1.1\r\n"
                    f"Host: {host}:{port}\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {key}\r\n"
                    "Sec-WebSocket-Version: 13\r\n\r\n"
                )
                sock.sendall(request.encode("ascii"))
                sock.recv(1024)

                _write_ws_frame(
                    sock,
                    json.dumps(
                        {
                            "text": "describe",
                            "attachments": [
                                {
                                    "filename": "image.png",
                                    "data": base64.b64encode(image.read_bytes()).decode(
                                        "ascii"
                                    ),
                                }
                            ],
                        }
                    ),
                )
                reply = json.loads(_read_ws_frame(sock))
        finally:
            server.shutdown()
            thread.join(timeout=1)

    assert reply == {"reply": "inline ws reply"}


def test_chat_handler_accepts_multipart_upload(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")

    calls: dict[str, object] = {}

    class BatchFeature(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ):
            calls["messages"] = messages
            import torch

            return BatchFeature({"input_ids": torch.tensor([[1, 2, 3]])})

        def batch_decode(self, outputs, skip_special_tokens=False):
            return ["multipart reply"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            import torch

            return torch.tensor([[1, 2, 3, 4, 5]])

    service = ChatService(
        model=FakeModel(),
        processor=FakeProcessor(),
        system_prompt="",
        prompt_style="auto",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
    )
    handler = make_handler(service)

    boundary = "----project-kagya-boundary"
    body = b"".join(
        [
            b"--" + boundary.encode("ascii") + b"\r\n",
            b'Content-Disposition: form-data; name="text"\r\n\r\n',
            b"describe this\r\n",
            b"--" + boundary.encode("ascii") + b"\r\n",
            b'Content-Disposition: form-data; name="attachments"; filename="image.png"\r\n',
            b"Content-Type: image/png\r\n\r\n",
            image.read_bytes(),
            b"\r\n--" + boundary.encode("ascii") + b"--\r\n",
        ]
    )

    from http.server import ThreadingHTTPServer
    from threading import Thread

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            request = Request(
                f"http://{host}:{port}/chat",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with closing(urlopen(request)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            thread.join(timeout=1)

    assert payload == {"reply": "multipart reply"}
    assert calls["messages"][0]["content"][0]["type"] == "image"
