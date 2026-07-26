# Chat Jobs, Streaming, and Cancellation

Issue #147 adds a versioned public `OperationStatus` contract shared by chat jobs and future action progress. Its terminal states are `completed`, `failed`, and `canceled`; status sequence numbers increase on every projection update, and only queued operations expose a queue position.

## Public API

- `POST /api/chat/jobs` returns `202` and requires `Idempotency-Key`. `X-KAGYA-Client-ID` or `client_session_id` scopes idempotency.
- `GET /api/chat/jobs/{operation_id}` returns public status.
- `GET /api/chat/jobs/{operation_id}/result` returns the final public `ChatResponse` after commit.
- `GET /api/chat/jobs/{operation_id}/events` emits SSE `status`, validated public `token`, `final`, `error`, and heartbeat frames. `Last-Event-ID` resumes bounded process-local replay.
- `DELETE /api/chat/jobs/{operation_id}` requests cooperative cancellation. Completed operations return `already_completed` and are never rewritten.
- `POST /api/chat` remains compatible and now enqueues and waits through the same authoritative path.

The durable registry stores public status/final results and an authenticated encrypted request spool needed to replay queued work. A validated public result may be staged during `finalizing`, but is exposed only after Journal commit evidence. It never persists generated partial text, built prompts, hidden thought, or raw model JSON. Public token events are split only from `visible_response` after AgentRuntime has committed the event.

## Recovery And Cancellation

The registry submits directly to the one `AgentRuntime`; there is no competing chat worker. Durable enqueue order therefore cannot contradict runtime processing order. On restart, queued records replay in enqueue order, interrupted running/finalizing records fail with `interrupted`, and committed final records remain available.

Queued cancellation reaches a checkpoint before chat mutation. Running providers receive a cancellation token; Transformers adds a stopping criterion, while unsupported providers finish generation privately and abort at the next checkpoint before persistence. Cancellation is also checked before Journal/WAL finalization. Timeout uses the same path with the bounded `timeout` code. Disconnecting SSE never cancels work.
