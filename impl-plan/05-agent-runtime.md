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
- Assign a strictly increasing processing sequence on that consumer. A missing snapshot starts from zero; a valid R04 snapshot restores the last successfully checkpointed sequence so the next event receives `N + 1`.
- Keep event metadata immutable and bounded to event identity, event type, constant source, request/acceptance time, and processing sequence. Request bodies, prompts, hidden/private reasoning, retrieved private memory, credentials, attachments, and arbitrary payloads must remain only in ephemeral in-memory handler closures and must not enter event metadata.
- Distinguish `submit -> accepted -> ordered -> executed`. Acceptance does not mean that execution, persistence, durability, or an external effect has completed.
- Reject submission without mutation when the queue is full or the runtime is not accepting.
- Once accepted, execute an event even if its caller stops waiting or cancels its result future.
- On shutdown, stop accepting first, drain accepted events, and then stop the consumer.
- Isolate handler failures so later accepted events still execute. R03 provides no transactional rollback: a handler that mutates state and then fails may leave a partial mutation.
- `AgentRuntime` is not persistence authority and never serializes files. It provides no queue persistence, replay, EventJournal, or StateWAL.

## AgentStateStore Requirements

- Treat `AgentStateStore` as the versioned snapshot persistence authority, separate from the `AgentRuntime` ordering/execution authority.
- Snapshot schema version 1 owns only a timezone-aware save time, the last successfully checkpointed processing sequence, and the current emotion values (`valence`, `arousal`, and `optimal_loss`).
- Restore the strict snapshot and EmotionState before `AgentRuntime` becomes accepting. A missing canonical file bootstraps the configured baseline; a corrupt, private, invalid, or unsupported existing file fails startup instead of becoming fresh state.
- Publish canonical JSON with a same-directory temporary file, mode `0600`, file flush/fsync, atomic replacement, and parent-directory fsync. Durable success is reported only after the directory fsync succeeds.
- Execute successful mutations in this order: `handler success -> capture EmotionState and sequence -> fsynced atomic snapshot checkpoint -> successful event outcome`.
- If the handler fails, do not checkpoint. If the handler succeeds but checkpointing fails, return a typed failure without claiming rollback or durable success, and keep the consumer available for later events.
- An accepted event that has not completed its snapshot checkpoint is not crash durable in R04. Queue contents and event IDs are never restored.
- Do not snapshot SessionState turns, chat transcripts, user messages, prompts, private reasoning, debug traces, request/event payloads, Memory records, or AdapterRegistry records. Memory and AdapterRegistry remain independent persistence authorities.
- Support only the strict historical v0-to-v1 migration. R05 and R06 still own EventJournal, StateWAL, exact crash classification, and deterministic reconstruction beyond this minimal snapshot.

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

## Completion Criteria

- Main loop integration test passes with no real model load.
- R03 and later AgentRuntime or persistence work must preserve this R02 boundary and must not make private reasoning durable or authoritative.
- R04 adds only the minimal EmotionState/sequence snapshot checkpoint; Journal and WAL layers must strengthen lifecycle evidence without redefining acceptance as persistence.
