# 05 Prompt Agent And Runtime Main Loop

## Goal

Connect prediction error, emotion, memory retrieval, prompt construction, response generation, postprocessing, and episode storage into one runtime loop.

## Target Files

- `kagya/persona/prompt_builder.py`
- `kagya/persona/conscious_agent.py`
- `kagya/runtime/__init__.py`
- `kagya/runtime/agent_runtime.py`
- `kagya/runtime/session_state.py`
- `kagya/runtime/main_loop.py`
- `tests/test_agent_runtime.py`
- `tests/test_main_loop.py`

## PromptBuilder Requirements

- Build prompts from user input, current emotion state, and retrieved memory context.
- Include valence, arousal, and optimal loss.
- Include related DB1 episodes.
- Include related DB2 semantic memories.
- If the model is instructed to produce `<think>...</think>` followed by the final answer, treat the private segment only as ephemeral raw generation input to postprocessing.
- Make clear that internal thought is neither user-visible nor durable/training authority.

## ConsciousAgent Requirements

- Wrap a `ModelProvider` and call `generate` with configured generation parameters.
- Keep generation provider-agnostic.
- Do not expose hidden thought filtering here; leave that to `ResponsePostProcessor`.

## Main Loop Requirements

- Treat `KagyaMainLoop` as the chat/cognition orchestration compatibility facade, not as the authority that orders concurrent subject mutations.
- Implement `KagyaMainLoop.chat(user_input: str) -> ChatResult` for ordinary chat; it never returns private/debug data.
- Implement the separate ephemeral diagnostic boundary `KagyaMainLoop.chat_debug(user_input: str) -> tuple[ChatResult, DebugChatTrace]`.
- Process in this order: input, context, surprisal, emotion update, memory retrieval, prompt build, generation, postprocess, DB1 save, result return.
- Store no hidden/private model reasoning in DB1 documents or metadata.
- Keep ordinary `ChatResult` limited to visible response and explicitly public structured data such as episode ID, loss/emotion values, model ID, and adapter ID; it does not own a hidden-thought field.
- When explicitly requested and authorized, expose private diagnostics through a separate request-scoped debug boundary that cannot be persisted or returned through the ordinary result contract.

## AgentRuntime Requirements

- Treat `AgentRuntime` as the single process-local authority for acceptance ordering and execution of authoritative subject mutations.
- Admit events non-blockingly to one bounded queue and execute accepted handlers in FIFO order on exactly one consumer thread.
- Assign a strictly increasing processing sequence on that consumer. R05 initializes from the Journal processing high-water, which can exceed the committed snapshot sequence after a failed event.
- Keep event metadata immutable and bounded to event identity, event type, constant source, request/acceptance time, and processing sequence. Request bodies, prompts, hidden/private reasoning, retrieved private memory, credentials, attachments, and arbitrary payloads must remain only in ephemeral in-memory handler closures and must not enter event metadata.
- Distinguish `submit -> accepted durable -> ordered -> started durable -> executed`. Acceptance proves only durable admission evidence, not execution, snapshot commit, or an external effect.
- Reject submission without mutation when the queue is full or the runtime is not accepting.
- Once accepted, execute an event even if its caller stops waiting or cancels its result future.
- On shutdown, stop accepting first, drain accepted events, and then stop the consumer.
- Isolate handler failures only after restoring R04-owned snapshot state and durably recording bounded failed evidence. This is not a general transaction or compensation mechanism.
- Enter explicit fail-stop on lifecycle/snapshot durability failure, reject admission, and fail pending work without executing later handlers.
- `AgentRuntime` is not persistence authority and never serializes files. It invokes narrow ordered lifecycle/checkpoint callbacks and provides no queue persistence, replay, or StateWAL.

## AgentStateStore Requirements

- Treat `AgentStateStore` as the versioned snapshot persistence authority, separate from the `AgentRuntime` ordering/execution authority.
- Snapshot schema version 1 owns only a timezone-aware save time, the last successfully checkpointed processing sequence, and the current emotion values (`valence`, `arousal`, and `optimal_loss`).
- Restore the strict snapshot and EmotionState before `AgentRuntime` becomes accepting. A missing canonical file bootstraps the configured baseline; a corrupt, private, invalid, or unsupported existing file fails startup instead of becoming fresh state.
- Publish canonical JSON with a same-directory temporary file, mode `0600`, file flush/fsync, atomic replacement, and parent-directory fsync. Durable success is reported only after the directory fsync succeeds.
- Execute successful R05 mutations in this order: `prepared durable -> fsynced atomic snapshot checkpoint -> completed durable -> successful event outcome`.
- On handler failure before prepared, restore only the last committed R04 snapshot. Do not roll back SessionState, Memory, AdapterRegistry, tools, network effects, or other authorities.
- If snapshot publication or terminal lifecycle evidence fails, return an indeterminate durability failure and fail-stop without rollback.
- An accepted event that has not completed its snapshot checkpoint is not crash durable in R04. Queue contents and event IDs are never restored.
- Do not snapshot SessionState turns, chat transcripts, user messages, prompts, private reasoning, debug traces, request/event payloads, Memory records, or AdapterRegistry records. Memory and AdapterRegistry remain independent persistence authorities.
- Support only the strict historical v0-to-v1 migration. R05 hashes the exact canonical snapshot bytes, while R06 still owns StateWAL and deterministic reconstruction.

## EventJournal Requirements

- Treat `EventJournal` as the durable lifecycle, integrity, processing high-water, and crash-classification authority; it is not a second state store.
- Persist strict metadata-only `accepted`, `started`, `prepared`, `completed`, `failed`, `recovery_classified`, and `checkpoint` records in a canonical SHA-256 chain.
- Under the runtime admission lock, check status/capacity, durably append accepted, then enqueue so concurrent durable acceptance and FIFO admission have one order.
- Append started after assigning sequence and before invoking the handler. Successful handlers append prepared with canonical before/after state hashes, publish only through `AgentStateStore`, then append completed before Future success.
- A handler failure consumes its sequence. After R04 restore and durable failed evidence, later events may continue from the Journal high-water even though the snapshot sequence remains older.
- Verify every retained segment and reconcile it with the canonical snapshot before runtime acceptance. Recovery classifies accepted-only, uncommitted started/prepared, and matching prepared-plus-snapshot outcomes without replaying handlers.
- Fail closed without truncation or repair on malformed/partial records, unsupported versions, hash or lifecycle breaks, sequence gaps, missing rotation artifacts, or Journal/snapshot mismatch.
- Require a private service-owned Journal directory and one exclusive cross-process Journal authority so concurrent processes cannot fork the hash chain.
- Rotate only at quiescent lifecycle boundaries and start each segment with a checkpoint preserving hash continuity, processing high-water, and snapshot identity.
- Persist no messages, transcripts, prompts, private reasoning, request/event payloads, attachments, credentials, raw exceptions, tracebacks, or private filesystem details. Public Journal/runtime durability errors remain typed and bounded.
- Do not introduce StateWAL, reconstruction, handler replay, external-store reconciliation, backup, encryption, or generalized rollback in R05.

## Test Requirements

- `DummyProvider` drives `user_input -> response` end-to-end.
- DB1 receives a saved episode.
- Visible response does not contain `<think>`.
- Ordinary `ChatResult` and normal API schemas have no private/debug field.
- Explicit debug inspection can observe private data only for the current request, and saving the same turn leaves no private sentinel in DB1.
- Emotion state changes after loss calculation.
- Concurrent mutation producers cannot bypass the single-consumer runtime, rejected queue-full work does not execute, shutdown drains accepted work, caller cancellation does not cancel accepted work, and one handler failure does not terminate the consumer.
- Serialized event metadata contains no private sentinel or arbitrary operation payload.
- AgentState tests prove strict schema/migration/privacy rejection, atomic replacement, file and directory fsync, mode `0600`, restore-before-acceptance, sequence continuation, and exclusion of SessionState/Memory/AdapterRegistry data.
- EventJournal tests prove durable lifecycle ordering, hash/lifecycle verification, Journal/snapshot reconciliation, failed-sequence high-water, fail-stop behavior, bounded rotation, restrictive permissions, and private-sentinel exclusion.

## Completion Criteria

- Main loop integration test passes with no real model load.
- R03 and later AgentRuntime or persistence work must preserve this R02 boundary and must not make private reasoning durable or authoritative.
- R05 adds durable lifecycle and crash-classification evidence without adding StateWAL reconstruction or becoming snapshot authority.
