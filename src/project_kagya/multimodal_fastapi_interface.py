from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from starlette.websockets import WebSocketDisconnect


@dataclass(slots=True)
class IngestSummary:
    received_text: str | None
    received_files: list[str]
    modalities: list[str]


@dataclass(slots=True)
class ChatSummary:
    message: str


@dataclass(slots=True)
class StreamSummary:
    status: str


class MultimodalIngestAPI:
    def __init__(self) -> None:
        self.app = FastAPI(title="PROJECT-KAGYA Multimodal Ingest API")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.post("/ingest")
        async def ingest(
            text: str | None = Form(default=None),
            image: UploadFile | None = File(default=None),
            audio: UploadFile | None = File(default=None),
            video: UploadFile | None = File(default=None),
        ) -> IngestSummary:
            files = [file for file in [image, audio, video] if file is not None]
            if text is None and not files:
                raise HTTPException(status_code=400, detail="no input provided")

            received_files = [file.filename or "" for file in files]
            modalities = self._collect_modalities(text, files)
            return IngestSummary(
                received_text=text,
                received_files=received_files,
                modalities=modalities,
            )

        @self.app.post("/chat")
        async def chat(message: str = Form(...)) -> ChatSummary:
            if not message.strip():
                raise HTTPException(status_code=400, detail="message must not be empty")
            return ChatSummary(message=message)

        @self.app.websocket("/stream")
        async def stream(websocket: WebSocket) -> None:
            await websocket.accept()
            try:
                while True:
                    message = await websocket.receive_text()
                    if not message.strip():
                        await websocket.send_json(
                            {"status": "error", "detail": "empty message"}
                        )
                        continue
                    await websocket.send_json({"status": "ok", "echo": message})
            except WebSocketDisconnect:
                return
            except Exception:
                await websocket.close()

    @staticmethod
    def _collect_modalities(text: str | None, files: list[UploadFile]) -> list[str]:
        modalities: list[str] = []
        if text is not None:
            modalities.append("text")
        for file in files:
            modalities.append(MultimodalIngestAPI._classify_file(file))
        return modalities

    @staticmethod
    def _classify_file(file: UploadFile) -> str:
        content_type = file.content_type or ""
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("audio/"):
            return "audio"
        if content_type.startswith("video/"):
            return "video"
        raise HTTPException(status_code=415, detail="unsupported media type")


def create_app() -> FastAPI:
    return MultimodalIngestAPI().app
