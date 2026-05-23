# 08 FastAPI Backend

## Goal

Expose PROJECT-KAGYA runtime, memory, sleep, and adapter controls through FastAPI while preserving debug/normal data boundaries.

## Target Files

- `kagya/api/server.py`
- `kagya/api/dependencies.py`
- `kagya/api/schemas/__init__.py`
- `kagya/api/schemas/chat.py`
- `kagya/api/schemas/debug.py`
- `kagya/api/schemas/memory.py`
- `kagya/api/schemas/adapter.py`
- `kagya/api/schemas/sleep.py`
- `kagya/api/routes/__init__.py`
- `kagya/api/routes/chat.py`
- `kagya/api/routes/debug.py`
- `kagya/api/routes/memory.py`
- `kagya/api/routes/sleep.py`
- `kagya/api/routes/adapters.py`

## Endpoint Requirements

- `POST /api/chat`
- `POST /api/chat/debug`
- `GET /api/state/emotion`
- `GET /api/memory/search`
- `GET /api/memory/episodes/{episode_id}`
- `GET /api/memory/semantic/{memory_id}`
- `POST /api/sleep/run`
- `GET /api/adapters`
- `POST /api/adapters/{adapter_id}/evaluate`
- `POST /api/adapters/{adapter_id}/trial`
- `POST /api/adapters/{adapter_id}/approve`
- `POST /api/adapters/{adapter_id}/activate`
- `POST /api/adapters/{adapter_id}/reject`

## Chat API Requirements

- Request schema includes `message`, `attachments`, and `debug`.
- Attachments are accepted as schema-only in v1.0; processing remains text-only.
- Normal `/api/chat` response includes `episode_id`, `response`, `emotion`, and `model`.
- Normal `/api/chat` response must not include `hidden_thought`, raw prompt, raw retrieved memory, or `<think>`.
- `/api/chat/debug` response may include `hidden_thought`, `loss`, `emotion`, retrieved memory, prompt, and generation params.

## Security And Boundary Requirements

- Allow CORS origins only from configuration.
- Treat debug endpoint as development-only and document future authentication requirement.
- Never merge debug fields into normal response schemas.

## Test Requirements

- `/api/chat` works with `DummyProvider`.
- `/api/chat` response does not contain hidden thought or think tags.
- `/api/chat/debug` includes hidden thought and loss.
- CORS middleware uses configured origins.
- Adapter endpoints enforce lifecycle transitions.
- Sleep endpoint returns sleep cycle result in dry-run mode.

## Completion Criteria

- FastAPI backend can serve the full DummyProvider flow.
