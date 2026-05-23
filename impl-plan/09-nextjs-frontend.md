# 09 Next.js Frontend

## Goal

Implement a frontend that communicates with the FastAPI backend and keeps normal chat separate from debug internals.

## Target Directory

- `frontend/`

## Technology Requirements

- Next.js.
- TypeScript.
- shadcn/ui.
- TanStack Query.
- OpenAPI client generation may be used if helpful.

## Required Pages

- `/chat`
- `/debug`
- `/memory`
- `/sleep`
- `/adapters`
- `/evaluations`

## Chat UI Requirements

- Show chat history.
- Show current model ID.
- Show active adapter.
- Show simple valence and arousal indicators.
- Do not show `hidden_thought`.
- Do not show raw prompt.
- Do not show raw retrieved memory.

## Debug UI Requirements

- Show hidden thought.
- Show raw prompt.
- Show loss.
- Show valence, arousal, and optimal loss.
- Show retrieved DB1 and DB2 memory.
- Show model ID and adapter ID.
- Show token and generation params when available.

## Adapter UI Requirements

- List adapters by status.
- Support evaluate, trial, approve, activate, reject, and rollback actions where backend supports them.
- Clearly distinguish candidate, trial active, approved, active, rejected, and archived states.

## Sleep UI Requirements

- Trigger sleep cycle.
- Show target episode count.
- Show dream dataset preview.
- Show adapter candidate creation result.

## Test Requirements

- `/chat` sends requests to `/api/chat`.
- `/chat` never renders hidden thought fields.
- `/debug` sends requests to `/api/chat/debug`.
- `/debug` renders debug-only fields.
- Adapter actions call the correct backend endpoints.
- Sleep page calls `/api/sleep/run` and displays dry-run result.

## Completion Criteria

- `/chat` and `/debug` communicate with FastAPI successfully.
- Debug information is not leaked into normal chat UI.
