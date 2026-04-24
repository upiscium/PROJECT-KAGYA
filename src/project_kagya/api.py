"""HTTP API for multimodal chat requests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import email.parser
import email.policy
import json
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from project_kagya import chat


@dataclass(frozen=True)
class ChatRequest:
    text: str
    attachments: tuple[chat.Attachment, ...]


@dataclass(frozen=True)
class ChatService:
    model: Any
    processor: Any
    system_prompt: str
    prompt_style: chat.PromptStyle
    max_new_tokens: int
    temperature: float
    top_p: float

    def reply(self, request: ChatRequest) -> str:
        messages = chat.build_messages(
            self.system_prompt,
            (),
            request.text,
            request.attachments,
        )

        if request.attachments:
            if not hasattr(self.processor, "apply_chat_template"):
                raise RuntimeError(
                    "Multimodal attachments require a processor with apply_chat_template support."
                )
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            input_ids = inputs["input_ids"]
            input_length = input_ids.shape[-1]
        else:
            inputs, input_length = chat._prepare_inputs(  # noqa: SLF001
                self.processor,
                messages,
                self.prompt_style,
            )

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
        )
        return chat._decode_outputs(self.processor, outputs, input_length)  # noqa: SLF001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve multimodal chat over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--backend", choices=["auto", "transformers"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument(
        "--prompt-style", choices=["auto", "chat", "plain"], default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser


def _infer_media_type(path: Path) -> chat.MediaType:
    return chat._infer_media_type(path)  # noqa: SLF001


def _build_attachment(record: Mapping[str, Any]) -> chat.Attachment:
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Each attachment must include a path.")

    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Attachment not found: {path_value}")

    media_type_value = record.get("media_type")
    if media_type_value is None:
        media_type = _infer_media_type(path)
    elif media_type_value in {"image", "audio", "video"}:
        media_type = media_type_value
    else:
        raise ValueError(f"Unsupported attachment type: {media_type_value}")

    name_value = record.get("name")
    name = path.name if name_value is None else str(name_value)
    return chat.Attachment(path=path.resolve(), media_type=media_type, name=name)


def _build_uploaded_attachment(
    filename: str,
    content: bytes,
    temp_dir: tempfile.TemporaryDirectory[str],
) -> chat.Attachment:
    path = Path(temp_dir.name) / filename
    path.write_bytes(content)
    media_type = _infer_media_type(path)
    return chat.Attachment(path=path.resolve(), media_type=media_type, name=path.name)


def _build_inline_attachment(
    record: Mapping[str, Any],
    temp_dir: tempfile.TemporaryDirectory[str],
) -> chat.Attachment:
    filename_value = record.get("filename")
    if not isinstance(filename_value, str) or not filename_value.strip():
        raise ValueError("Each inline attachment must include a filename.")

    payload_value = record.get("data")
    if not isinstance(payload_value, str) or not payload_value.strip():
        raise ValueError("Each inline attachment must include base64 data.")

    try:
        content = base64.b64decode(payload_value, validate=True)
    except Exception as error:  # noqa: BLE001
        raise ValueError("Inline attachment data must be valid base64.") from error

    filename = Path(filename_value).name
    path = Path(temp_dir.name) / filename
    path.write_bytes(content)

    media_type_value = record.get("media_type")
    if media_type_value is None:
        media_type = _infer_media_type(path)
    elif media_type_value in {"image", "audio", "video"}:
        media_type = media_type_value
    else:
        raise ValueError(f"Unsupported attachment type: {media_type_value}")

    return chat.Attachment(path=path.resolve(), media_type=media_type, name=path.name)


def parse_multipart_chat_request(
    content_type: str, body: bytes
) -> tuple[ChatRequest, tuple[tempfile.TemporaryDirectory[str], ...]]:
    parser = email.parser.BytesParser(policy=email.policy.default)
    message = parser.parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    if not message.is_multipart():
        raise ValueError("multipart/form-data body is malformed.")

    text = ""
    attachments: list[chat.Attachment] = []
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name == "text" and part.get_filename() is None:
            text = part.get_content().strip()
            continue
        if name == "attachments" and part.get_filename():
            temp_dir = tempfile.TemporaryDirectory(prefix="project-kagya-upload-")
            temp_dirs.append(temp_dir)
            attachments.append(
                _build_uploaded_attachment(
                    Path(part.get_filename()).name,
                    part.get_payload(decode=True) or b"",
                    temp_dir,
                )
            )
            continue
        if name:
            raise ValueError(f"Unsupported multipart field: {name}")

    if not text and not attachments:
        raise ValueError("A request must include text or attachments.")
    return ChatRequest(text=text, attachments=tuple(attachments)), tuple(temp_dirs)


def parse_websocket_chat_request(
    payload: Mapping[str, Any],
) -> tuple[ChatRequest, tuple[tempfile.TemporaryDirectory[str], ...]]:
    text_value = payload.get("text", "")
    if not isinstance(text_value, str):
        raise ValueError("'text' must be a string.")
    text = text_value.strip()

    attachments_value = payload.get("attachments", [])
    if not isinstance(attachments_value, list):
        raise ValueError("'attachments' must be a list.")

    attachments: list[chat.Attachment] = []
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    for record in attachments_value:
        if not isinstance(record, Mapping):
            raise ValueError("Each attachment must be an object.")

        if "data" in record:
            temp_dir = tempfile.TemporaryDirectory(prefix="project-kagya-ws-upload-")
            temp_dirs.append(temp_dir)
            attachments.append(_build_inline_attachment(record, temp_dir))
            continue

        attachments.append(_build_attachment(record))

    if not text and not attachments:
        raise ValueError("A request must include text or attachments.")
    return ChatRequest(text=text, attachments=tuple(attachments)), tuple(temp_dirs)


def parse_chat_request(payload: Mapping[str, Any]) -> ChatRequest:
    text_value = payload.get("text", "")
    if not isinstance(text_value, str):
        raise ValueError("'text' must be a string.")
    text = text_value.strip()

    attachments_value = payload.get("attachments", [])
    if not isinstance(attachments_value, list):
        raise ValueError("'attachments' must be a list.")

    attachments = tuple(_build_attachment(record) for record in attachments_value)
    if not text and not attachments:
        raise ValueError("A request must include text or attachments.")
    return ChatRequest(text=text, attachments=attachments)


def _json_response(
    status: HTTPStatus, payload: Mapping[str, Any]
) -> tuple[HTTPStatus, bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, body


def _is_websocket_upgrade(handler: BaseHTTPRequestHandler) -> bool:
    upgrade = handler.headers.get("Upgrade", "")
    connection = handler.headers.get("Connection", "")
    return upgrade.lower() == "websocket" and "upgrade" in connection.lower()


def _handshake_websocket(handler: BaseHTTPRequestHandler) -> bool:
    key = handler.headers.get("Sec-WebSocket-Key", "")
    if not key:
        return False

    accept_source = (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("utf-8")
    accept = base64.b64encode(hashlib.sha1(accept_source).digest()).decode("ascii")
    handler.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()
    return True


def _read_exact(stream, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = stream.read(length - len(data))
        if not chunk:
            raise ConnectionError("unexpected websocket close")
        data.extend(chunk)
    return bytes(data)


def _read_websocket_message(handler: BaseHTTPRequestHandler) -> str:
    stream = handler.rfile
    first_byte = _read_exact(stream, 1)[0]
    fin = bool(first_byte & 0x80)
    opcode = first_byte & 0x0F
    if opcode == 0x8:
        raise ConnectionError("websocket closed")
    if opcode != 0x1 or not fin:
        raise ValueError("Only single-frame text websocket messages are supported.")

    second_byte = _read_exact(stream, 1)[0]
    masked = bool(second_byte & 0x80)
    length = second_byte & 0x7F
    if length == 126:
        length = int.from_bytes(_read_exact(stream, 2), "big")
    elif length == 127:
        length = int.from_bytes(_read_exact(stream, 8), "big")

    mask = _read_exact(stream, 4) if masked else b"\x00\x00\x00\x00"
    payload = bytearray(_read_exact(stream, length))
    if masked:
        for index, value in enumerate(payload):
            payload[index] = value ^ mask[index % 4]
    return payload.decode("utf-8")


def _write_websocket_message(handler: BaseHTTPRequestHandler, message: str) -> None:
    payload = message.encode("utf-8")
    frame = bytearray([0x81])
    length = len(payload)
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(length.to_bytes(2, "big"))
    else:
        frame.append(127)
        frame.extend(length.to_bytes(8, "big"))
    frame.extend(payload)
    handler.connection.sendall(frame)


def _websocket_error(payload: str) -> str:
    return json.dumps({"error": payload}, ensure_ascii=False)


def make_handler(service: ChatService):
    class ChatRequestHandler(BaseHTTPRequestHandler):
        server_version = "project-kagya-api/0.1"

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/ws" and _is_websocket_upgrade(self):
                if not _handshake_websocket(self):
                    self._send_json(
                        HTTPStatus.BAD_REQUEST, {"error": "missing websocket key"}
                    )
                    return
                while True:
                    try:
                        payload = _read_websocket_message(self)
                        request, temp_dirs = parse_websocket_chat_request(
                            json.loads(payload)
                        )
                        reply = service.reply(request)
                        _write_websocket_message(
                            self, json.dumps({"reply": reply}, ensure_ascii=False)
                        )
                        for temp_dir in temp_dirs:
                            temp_dir.cleanup()
                    except ConnectionError:
                        return
                    except FileNotFoundError as error:
                        _write_websocket_message(self, _websocket_error(str(error)))
                    except ValueError as error:
                        _write_websocket_message(self, _websocket_error(str(error)))
                    except RuntimeError as error:
                        _write_websocket_message(self, _websocket_error(str(error)))
                return
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/chat":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", "0"))
            temp_dirs: tuple[tempfile.TemporaryDirectory[str], ...] = ()

            try:
                body = self.rfile.read(content_length)
                if "multipart/form-data" in content_type:
                    request, temp_dirs = parse_multipart_chat_request(
                        content_type, body
                    )
                elif "application/json" in content_type:
                    request = parse_chat_request(json.loads(body.decode("utf-8")))
                    temp_dirs = ()
                else:
                    self._send_json(
                        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        {
                            "error": "Content-Type must be application/json or multipart/form-data"
                        },
                    )
                    return
                reply = service.reply(request)
            except FileNotFoundError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except RuntimeError as error:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                return
            finally:
                for temp_dir in temp_dirs:
                    temp_dir.cleanup()

            self._send_json(HTTPStatus.OK, {"reply": reply})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    return ChatRequestHandler


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model, processor, _backend = chat._load_model_and_processor(args)  # noqa: SLF001
    service = ChatService(
        model=model,
        processor=processor,
        system_prompt=args.system_prompt,
        prompt_style=args.prompt_style,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    handler = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
